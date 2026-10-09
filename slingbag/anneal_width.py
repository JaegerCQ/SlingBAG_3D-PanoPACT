"""Propose narrower fine Gaussians while preserving each source's integral mass.

This deliberately changes Gaussian covariance, the spatial field and RF.
It is a new proposal whose step-zero metrics must be evaluated before training.
"""
import argparse,json
import numpy as np
import torch
from .io_utils import resolve_input,resolve_output
from .paths import is_artifact_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--fine-cutoff-mm',type=float,default=.2)
    parser.add_argument('--new-sigma-mm',type=float,default=.1)
    args=parser.parse_args()
    if args.new_sigma_mm<=0:raise ValueError('Width must be positive')
    checkpoint_path=resolve_input(args.checkpoint)
    if not is_artifact_path(checkpoint_path):raise ValueError('Checkpoint must reside in outputs or examples')
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if not saved['configuration'].get('random_only_ancestry'):raise ValueError('Requires random ancestry')
    s=saved['radius'].clone();p=saved['pressure'].clone()
    fine=s<args.fine_cutoff_mm*.001
    initial_mass=float((p.double()*s.double()**3).sum())
    old=s[fine].clone();s[fine]=args.new_sigma_mm*.001
    p[fine]*=(old/s[fine])**3
    final_mass=float((p.double()*s.double()**3).sum())
    error=abs(final_mass/initial_mass-1);assert error<1e-6
    record={'parent_checkpoint':str(resolve_input(args.checkpoint)),
        'fine_sources_changed':int(fine.sum()),'new_sigma_m':args.new_sigma_mm*.001,
        'mass_relative_error':error,'covariance_preserved':False,
        'purpose':'Test whether narrower random-derived Gaussians recover more coherent fine detail'}
    saved['pressure']=p;saved['radius']=s
    saved['configuration']={**saved['configuration'],'width_anneal':record}
    saved['record']=record
    output=resolve_output(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(saved,output);output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
