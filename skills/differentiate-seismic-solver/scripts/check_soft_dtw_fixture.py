#!/usr/bin/env python3
"""Small CPU data-objective checks; never a target-solver acceptance certificate."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from soft_dtw_objective import PreparedSoftDTWDivergence, soft_dtw_cost


def check(backend):
    t=torch.arange(33,dtype=torch.float64)*.003
    observed=torch.stack([torch.sin(2*torch.pi*5*t),.4*torch.cos(2*torch.pi*4*t)])
    predicted=(.8*observed+.03*torch.sin(2*torch.pi*8*t)).requires_grad_(True)
    config=dict(dt=.003,gamma=.1,scale=.7,decimation=8,
                anti_alias_cutoff_hz=10.,anti_alias_transition_hz=2.)
    reference=PreparedSoftDTWDivergence(observed,backend='reference',**config)
    objective=PreparedSoftDTWDivergence(observed,backend=backend,**config)
    loss=objective(predicted)
    gradient,=torch.autograd.grad(loss,predicted)
    reference_loss=reference(predicted)
    reference_gradient,=torch.autograd.grad(reference_loss,predicted)
    torch.testing.assert_close(loss,reference_loss,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(gradient,reference_gradient,atol=1e-12,rtol=1e-12)
    direction=torch.cos(torch.arange(predicted.numel(),dtype=torch.float64)).reshape(predicted.shape)
    dot=(gradient*direction).sum()
    if dot.abs()<1e-5:raise ValueError('Uninformative data direction')
    finite=[]
    for h in [1e-4,5e-5,2.5e-5]:
        fd=(objective(predicted+h*direction)-objective(predicted-h*direction))/(2*h)
        torch.testing.assert_close(fd,dot,rtol=2e-6,atol=1e-9)
        finite.append(dict(step=h,relative_error=float((fd-dot).abs()/torch.maximum(fd.abs(),dot.abs()))))
    p=torch.tensor([[.2,-.3,.5,.9]],dtype=torch.float64,requires_grad=True)
    full=soft_dtw_cost(p,p,gamma=.2,backend=backend)
    one=soft_dtw_cost(p,p.detach(),gamma=.2,backend=backend)
    gfull,=torch.autograd.grad(full.sum(),p)
    gone,=torch.autograd.grad(one.sum(),p)
    torch.testing.assert_close(gfull,2*gone,rtol=1e-12,atol=1e-12)
    if torch.allclose(gfull,gone):raise ValueError('Self-term check is uninformative')
    return dict(status='passed',backend=backend,device='cpu',dtype='float64',
        scope='Small data-only reference/value/VJP checks, not native model or inversion acceptance',
        finite_difference=finite,predicted_self_two_arguments_checked=True,
        fixture_source_sha256=hashlib.sha256(Path(__file__).with_name('soft_dtw_objective.py').read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['reference','pysdtw'],default='reference')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=check(args.backend)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
