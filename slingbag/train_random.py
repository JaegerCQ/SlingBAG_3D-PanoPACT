"""Reconstruct measured RF from random Gaussians, without an image initializer.

Training reads only RF, sensor coordinates and the fixed sensor split. It never
reads references/, PA checkpoints, DAS images, or a previously fitted response.
"""
from pathlib import Path
import argparse
import json
import time
import hashlib

import numpy as np
import torch

from .rapid_forward import gaussian_forward
from .io_utils import data_path, resolve_input, resolve_output
from .random_prior import random_initialize, GaussianGridField, spatial_penalty
from .paths import is_artifact_path, response_path, display_path, dataset_root
from .dataset import load_dataset


def predict_signal(sensors, xyz, pressure, sigma, forward_kwargs, response=None):
    """Use the dataset's acquisition parameters for both training and evaluation."""
    value=gaussian_forward(sensors.contiguous(),xyz.contiguous(),pressure.contiguous(),
                           sigma.contiguous(),**forward_kwargs)
    if response is not None:
        value=torch.nn.functional.conv1d(value[:,None],response,padding=response.numel()//2)[:,0]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--balls',type=int,default=1000000)
    parser.add_argument('--seed',type=int,default=1013)
    parser.add_argument('--sigma-mm',type=float,default=.25)
    parser.add_argument('--initial-pressure',type=float,default=4.)
    parser.add_argument('--iterations',type=int,default=700)
    parser.add_argument('--batch-sensors',type=int,default=128)
    parser.add_argument('--field-grid',type=int,default=160)
    parser.add_argument('--prior-weight',type=float,default=1.)
    parser.add_argument('--sparsity-weight',type=float,default=.02)
    parser.add_argument('--curvature-weight',type=float,default=2.)
    parser.add_argument('--pressure-lr',type=float,default=.03)
    parser.add_argument('--adam-eps',type=float,default=1e-10)
    parser.add_argument('--position-lr',type=float,default=.003)
    parser.add_argument('--geometry-start',type=int,default=1000000)
    parser.add_argument('--max-shift-mm',type=float,default=.15)
    parser.add_argument('--initial-checkpoint',default=None)
    parser.add_argument('--all-sensors',action='store_true')
    parser.add_argument('--response-kernel',default=None)
    parser.add_argument('--eval-every',type=int,default=100)
    args=parser.parse_args()
    if args.iterations<1 or args.prior_weight<0 or args.sparsity_weight<0:
        raise ValueError('Invalid optimization settings')
    dataset=load_dataset()
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed+1)
    output=resolve_output(args.output);output.mkdir(parents=True,exist_ok=True)
    if args.initial_checkpoint:
        initial_path=resolve_input(args.initial_checkpoint)
        if not is_artifact_path(initial_path):
            raise ValueError('Continuation checkpoints must belong to this random experiment')
        initial=torch.load(initial_path,map_location='cpu',weights_only=False)
        if not initial.get('configuration',{}).get('random_only_ancestry'):
            raise ValueError('Checkpoint must have random-only ancestry')
        xyz0=initial['xyz'].numpy();p0=initial['pressure'].numpy();sigma0=initial['radius'].numpy()
        initialization={'type':'continuation of random-only point cloud','path':str(initial_path),
                        'sha256':hashlib.sha256(initial_path.read_bytes()).hexdigest()}
    else:
        xyz0,p0,sigma0,initialization=random_initialize(args.balls,sigma=args.sigma_mm*.001,
            pressure=args.initial_pressure,seed=args.seed)
        # The amplitudes, too, are independent of the measured image/signal.
        p0=p0*rng.uniform(.5,1.5,len(p0)).astype(np.float32)
        initialization['pressure_multiplier']='uniform [0.5,1.5], separate RNG seeded with seed+1'
    xyz0=np.asarray(xyz0,dtype=np.float32);p0=np.asarray(p0,dtype=np.float32)
    sigma0=np.asarray(sigma0,dtype=np.float32)
    anchor=torch.as_tensor(xyz0,device='cuda')
    sigma=torch.as_tensor(sigma0,device='cuda')
    pressure=torch.nn.Parameter(torch.as_tensor(p0/100.,device='cuda'))
    offsets=torch.nn.Parameter(torch.zeros_like(anchor),requires_grad=args.geometry_start==0)
    sigma_values=np.unique(sigma0)
    if len(sigma_values)>4:raise ValueError('Use at most 4 fixed width groups for the grid prior')
    grid_spacing=.04/args.field_grid
    origin=(-.02+grid_spacing/2,)*3
    priors=[]
    for width in sigma_values:
        index=torch.as_tensor(np.flatnonzero(sigma0==width),device='cuda')
        module=GaussianGridField(sigma=float(width),dims=(args.field_grid,)*3,
            spacing=grid_spacing,origin=origin,compensate_cic=True).cuda()
        cache=module.prepare_deposition(anchor[index])
        priors.append((module,index,cache))
    target=torch.from_numpy(dataset.signal).cuda()
    sensors=torch.from_numpy(dataset.sensors).cuda()
    train_ids=np.arange(dataset.num_sensors) if args.all_sensors else dataset.train
    val_ids=dataset.validation
    energy=target[train_ids].square().mean().detach()
    response=None
    response_coefficients=None
    if args.response_kernel:
        path=response_path(args.response_kernel)
        h=np.load(path)
        response_coefficients=h.tolist()
        response=torch.tensor(h[::-1].copy(),device='cuda',dtype=torch.float32).view(1,1,-1)
    def predict(ids,xyz,p):
        return predict_signal(sensors[ids],xyz,p,sigma,dataset.forward_kwargs,response)
    config={**vars(args),'random_only_ancestry':True,'initialization':initialization,
        'ball_count':len(p0),'sigma_values_m':sigma_values.tolist(),
        'pressure_parameter_unit':100.,'position_parameter_unit_m':.0001,
        'field_spacing_m':grid_spacing,'field_amplitude_scale':100.,
        'field_modules':[module.metadata for module,_,_ in priors],
        'prior':'CIC Gaussian field: isotropic TV + curvature; no image target',
        'reference_images_used_for_training':False,
        'training_sensor_count':len(train_ids),'reserved_sensor_count':0 if args.all_sensors else len(val_ids),
        'validation_used_for_tuning':True,'validation_used_in_training':args.all_sensors,
        'response_kernel_coefficients':response_coefficients,
        'acquisition':{**dataset.forward_kwargs,'num_sensors':dataset.num_sensors,
                       't0_seconds':0.,'global_gain':dataset.gain},
        'selection':'minimum reserved-sensor MSE; when all_sensors, use fixed final iteration',
        'dataset_directory':display_path(dataset.directory),
        'input_files':{display_path(data_path(name)):hashlib.sha256(data_path(name).read_bytes()).hexdigest()
                       for name in ['signal_normalized.npy','sensor_locations.npy','split.npz','metadata.json']}}
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    optimizer=torch.optim.Adam([{'params':[pressure],'lr':args.pressure_lr},
                                {'params':[offsets],'lr':0.}],eps=args.adam_eps)
    best=None;history=[];permutation=np.array([],dtype=np.int64);cursor=0
    torch.cuda.synchronize();started=time.perf_counter()
    def snapshot(record):
        return {'xyz':(anchor+offsets*.0001).detach().cpu().clone(),
                'pressure':(pressure*100).detach().cpu().clone(),'radius':sigma.detach().cpu().clone(),
                'record':record,'configuration':config}
    with (output/'training.jsonl').open('w') as log:
        for step in range(args.iterations+1):
            if step==args.geometry_start:offsets.requires_grad_(True)
            xyz=anchor+offsets*.0001;p=pressure*100
            geometry_active=step>=args.geometry_start
            field=None
            for module,index,cache in priors:
                value=module(xyz[index],p[index],cache=None if geometry_active else cache)
                field=value if field is None else field+value
            penalty=spatial_penalty(field,spacing=grid_spacing,reference_spacing=.0001,
                                    amplitude_scale=100.,curvature_weight=args.curvature_weight)
            sparsity=field.mean()/100.
            if step%args.eval_every==0 or step==args.iterations:
                with torch.no_grad():
                    train_sse=0.
                    for start in range(0,len(train_ids),128):
                        ids=train_ids[start:start+128]
                        train_sse+=float((predict(ids,xyz,p)-target[ids]).double().square().sum())
                    train_mse=train_sse/(len(train_ids)*target.shape[1])
                    val_mse=float((predict(val_ids,xyz,p)-target[val_ids]).double().square().mean())
                record={'step':step,'train_full_mse':train_mse,
                    'reserved_or_original_subset_mse':val_mse,'subset_is_trained':args.all_sensors,
                    'spatial_penalty':float(penalty['loss'].detach()),'tv':float(penalty['tv'].detach()),
                    'curvature':float(penalty['curvature'].detach()),'mean_field':float(field.mean().detach()),
                    'nonzero_balls':int((pressure>0).sum()),'max_peak_pressure':float(p.max().detach()),'ball_count':len(p0),
                    'elapsed_seconds':time.perf_counter()-started}
                history.append(record);log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
                if best is None or val_mse<best['record']['reserved_or_original_subset_mse']:
                    best=snapshot(record);torch.save(best,output/'best_validation.pt')
                if step==0:torch.save(snapshot(record),output/'initial.pt')
                # Keep a preview checkpoint for intermediate rendering without stopping training.
                torch.save(snapshot(record),output/'latest.pt')
                np.savez(output/'prior_field_mips.npz',xy=field.detach().amax(0).cpu().numpy(),
                    xz=field.detach().amax(1).cpu().numpy(),yz=field.detach().amax(2).cpu().numpy(),step=step)
            if step==args.iterations:break
            if cursor+args.batch_sensors>len(permutation):
                permutation=rng.permutation(train_ids);cursor=0
            ids=permutation[cursor:cursor+args.batch_sensors];cursor+=args.batch_sensors
            prediction=predict(ids,xyz,p)
            data_loss=(prediction-target[ids]).square().mean()/energy
            loss=data_loss+args.prior_weight*penalty['loss']+args.sparsity_weight*sparsity
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite objective at {step}')
            decay=.2+.8*.5*(1+np.cos(np.pi*step/args.iterations))
            optimizer.param_groups[0]['lr']=args.pressure_lr*decay
            optimizer.param_groups[1]['lr']=args.position_lr*decay if geometry_active else 0.
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if step==0:
                quantiles=torch.quantile(pressure.grad.detach().abs(),pressure.new_tensor([0.,.5,.95,1.])).cpu().tolist()
                (output/'initial_gradient.json').write_text(json.dumps({'abs_pressure_parameter_gradient_quantiles_0_50_95_100':quantiles,
                    'adam_eps':args.adam_eps,'pressure_parameter_unit':100.},indent=2)+'\n')
            optimizer.step()
            with torch.no_grad():
                pressure.clamp_(min=0.)
                norm=torch.linalg.vector_norm(offsets,dim=1).clamp_min(1e-20)
                offsets.mul_((args.max_shift_mm*10/norm).clamp(max=1)[:,None])
                offsets.copy_(((anchor+offsets*.0001).clamp(-.02,.02)-anchor)/.0001)
    final=snapshot(history[-1]);torch.save(final,output/'final.pt')
    torch.save(final if args.all_sensors else best,output/'selected.pt')
    summary={'configuration':config,'initial':history[0],'best_validation':best['record'],
             'final':history[-1],'selected':final['record'] if args.all_sensors else best['record'],
             'elapsed_seconds':time.perf_counter()-started}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Saved',output,flush=True)


if __name__=='__main__':main()
