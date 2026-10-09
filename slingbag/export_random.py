"""Export random-only Gaussian reconstruction on the original 400^3 grid."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.io import savemat
from .rapid_forward import gaussian_forward
from .voxelize_rapid import gaussian_voxelize
from .io_utils import data_path, resolve_input,resolve_output,save_ply
from .paths import is_artifact_path
from .dataset import load_dataset


def predict_acoustic(sensors, xyz, pressure, sigma, forward_kwargs):
    """Export uses the same metadata-driven sampling as the training stage."""
    return gaussian_forward(sensors,xyz,pressure,sigma,**forward_kwargs)


def validate_checkpoint_dataset(configuration, dataset):
    """Reject incompatible sampling/data while allowing legacy metadata records.

    File locations may change when moving a run. Signal, geometry and split
    hashes identify their contents; metadata acquisition fields identify the
    physical model without depending on provenance timestamps or old paths.
    """
    expected={**dataset.forward_kwargs,'num_sensors':dataset.num_sensors,
              't0_seconds':0.,'global_gain':dataset.gain}
    saved=configuration.get('acquisition')
    if saved is not None:
        for key,value in expected.items():
            old=saved.get(key)
            if old is None or not np.isfinite(old) or not np.isclose(old,value,rtol=1e-12,atol=0.):
                raise ValueError(f'Checkpoint acquisition {key}={old!r} differs from dataset value {value!r}')
    hashes=configuration.get('input_files',{})
    for name in ('signal_normalized.npy','sensor_locations.npy','split.npz'):
        known=[digest for path,digest in hashes.items() if Path(path).name==name]
        if known:
            current=hashlib.sha256((dataset.directory/name).read_bytes()).hexdigest()
            if any(value!=current for value in known):
                raise ValueError(f'Checkpoint input hash differs from the selected dataset: {name}')


def metrics(target,prediction):
    y=target.astype(np.float64).ravel();p=prediction.astype(np.float64).ravel()
    return {'mse':float(np.mean((y-p)**2)),'relative_l2':float(np.linalg.norm(y-p)/np.linalg.norm(y)),
            'correlation':float(np.corrcoef(y,p)[0,1])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--mat',action='store_true',help='Also export volume_zyx.mat (duplicates the large NPY volume)')
    args=parser.parse_args()
    dataset=load_dataset()
    folder=resolve_output(args.output);folder.mkdir(parents=True,exist_ok=True)
    checkpoint_path=resolve_input(args.checkpoint)
    if not is_artifact_path(checkpoint_path):raise ValueError('Checkpoint must reside in outputs or examples')
    ckpt=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if not ckpt['configuration']['random_only_ancestry']:raise ValueError('Not random-only')
    validate_checkpoint_dataset(ckpt['configuration'],dataset)
    keep=ckpt['pressure']>0
    xyz,p,s=[ckpt[k][keep].contiguous().cuda() for k in ('xyz','pressure','radius')]
    target=dataset.signal
    sensors=torch.from_numpy(dataset.sensors).cuda()
    started=time.perf_counter()
    with torch.no_grad():
        acoustic=predict_acoustic(sensors,xyz,p,s,dataset.forward_kwargs)
        h=ckpt['configuration'].get('response_kernel_coefficients')
        if h is not None:
            kernel=torch.tensor(h[::-1],device='cuda',dtype=torch.float32).view(1,1,-1)
            predicted=torch.nn.functional.conv1d(acoustic[:,None],kernel,padding=len(h)//2)[:,0].cpu().numpy()
        else:predicted=acoustic.cpu().numpy()
        volume=gaussian_voxelize(xyz,p,s,dims=(400,400,400),resolution=.0001,origin=(-.01995,)*3).cpu().numpy()
    np.save(folder/'acoustic_signal_normalized.npy',acoustic.cpu().numpy())
    np.save(folder/'predicted_signal_normalized.npy',predicted)
    gain=dataset.gain
    np.save(folder/'predicted_signal_original_units.npy',predicted/gain)
    np.save(folder/'volume_zyx.npy',volume)
    if args.mat:
        savemat(folder/'volume_zyx.mat',{'volume_zyx':volume,'axis_order':'ZYX','spacing_m':.0001,
                                       'origin_xyz_m':[-.01995]*3,'signal_gain':gain})
    save_ply(folder/'point_cloud.ply',xyz,p,s)
    torch.save({**ckpt,**{key:value.cpu() for key,value in zip(('xyz','pressure','radius'),(xyz,p,s))}},folder/'checkpoint.pt')
    report={'configuration':ckpt['configuration'],'record':ckpt['record'],
        'acquisition':{**dataset.forward_kwargs,'num_sensors':dataset.num_sensors,
                       't0_seconds':0.,'global_gain':dataset.gain},
        'source_checkpoint':str(resolve_input(args.checkpoint)),
        'source_checkpoint_sha256':hashlib.sha256(resolve_input(args.checkpoint).read_bytes()).hexdigest(),
        'candidate_balls':len(keep),
        'nonzero_balls':int(keep.sum()),'full':metrics(target,predicted),
        'original_train_subset':metrics(target[dataset.train],predicted[dataset.train]),
        'original_validation_subset':metrics(target[dataset.validation],predicted[dataset.validation]),
        'all_sensors_trained':ckpt['configuration']['all_sensors'],
        'scope':'Reserved sensors were used for tuning. In all-sensor runs they are also trained.',
        'elapsed_seconds':time.perf_counter()-started}
    (folder/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['candidate_balls','nonzero_balls','full','original_validation_subset']}),flush=True)


if __name__=='__main__':main()
