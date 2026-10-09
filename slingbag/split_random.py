"""Refine strong random-derived sources with mass/centroid/covariance conservation."""
import argparse,itertools,json
import numpy as np
import torch
from .io_utils import resolve_input,resolve_output
from .paths import is_artifact_path


def moments(x,p,s):
    mass=p.astype(np.float64)*s.astype(np.float64)**3
    xyz=x.astype(np.float64);total=mass.sum()
    centroid=np.einsum('i,ij->j',mass,xyz)/total
    second=np.einsum('i,ij,ik->jk',mass,xyz,xyz)/total+np.eye(3)*np.dot(mass,s.astype(np.float64)**2)/total
    return {'mass':float(total),'centroid':centroid.tolist(),'second_moment':second.tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--top',type=int,default=60000);parser.add_argument('--ratio',type=float,default=.6)
    args=parser.parse_args()
    if not 0<args.ratio<1 or args.top<1:raise ValueError('Invalid split settings')
    checkpoint_path=resolve_input(args.checkpoint)
    if not is_artifact_path(checkpoint_path):raise ValueError('Checkpoint must reside in outputs or examples')
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if not saved['configuration'].get('random_only_ancestry'):raise ValueError('Requires random-only ancestry')
    x,p,s=[saved[k].numpy() for k in ('xyz','pressure','radius')]
    shift=s*np.sqrt(1-args.ratio**2)
    eligible=np.flatnonzero((p>0)&np.all(np.abs(x)+shift[:,None]<.02,axis=1))
    mass=p[eligible].astype(np.float64)*s[eligible].astype(np.float64)**3
    if len(eligible)<=args.top:chosen=eligible
    else:chosen=eligible[np.argpartition(mass,-args.top)[-args.top:]]
    mask=np.ones(len(p),dtype=bool);mask[chosen]=False
    corners=np.asarray(list(itertools.product([-1.,1.],repeat=3)),dtype=np.float32)
    children=(x[chosen,None,:]+shift[chosen,None,None]*corners[None]).reshape(-1,3)
    widths=np.repeat(s[chosen]*args.ratio,8)
    amplitudes=np.repeat(p[chosen]/(8*args.ratio**3),8)
    newx=np.concatenate([x[mask],children]);newp=np.r_[p[mask],amplitudes];news=np.r_[s[mask],widths]
    before=moments(x,p,s);after=moments(newx,newp,news)
    mass_error=abs(after['mass']/before['mass']-1)
    centroid_error=float(np.max(np.abs(np.asarray(after['centroid'])-before['centroid'])))
    covariance_error=float(np.max(np.abs(np.asarray(after['second_moment'])-before['second_moment'])))
    assert mass_error<2e-6 and centroid_error<1e-8 and covariance_error<1e-9
    record={'parent_checkpoint':str(resolve_input(args.checkpoint)),'split_parents':len(chosen),
        'before_count':len(p),'after_count':len(newp),'width_ratio':args.ratio,
        'selection':'largest P*sigma^3 among nonnegative in-bounds random-derived sources',
        'mass_relative_error':mass_error,'centroid_max_absolute_error_m':centroid_error,
        'second_moment_max_absolute_error_m2':covariance_error,
        'conservation':'full Gaussian mass, centroid and second moment tensor; higher moments may change',
        'before_moments':before,'after_moments':after}
    config={**saved['configuration'],'random_only_ancestry':True,'adaptive_split':record}
    output=resolve_output(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'xyz':torch.from_numpy(newx),'pressure':torch.from_numpy(newp),
        'radius':torch.from_numpy(news),'configuration':config,'record':record},output)
    output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
