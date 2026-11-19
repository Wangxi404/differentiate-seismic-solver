"""Float64 data-domain soft-DTW divergence for one seismic shot.

The divergence follows Blondel, Mensch and Vert (2021), AISTATS130:3853--3861.
The optional fast first-order backend reuses the MIT-licensed dynamic-program
kernels of pysdtw0.0.5, with an independent precision-preserving Torch wrapper.
It avoids that package's torch.Tensor(numpy)/FloatTensor scalar round trips.

All leading dimensions represent traces of ONE shot; callers perform the
single shot reduction. Propagation is never called here. Preprocessing is the
declared zero-padded FFT antialias filter followed by fixed time sampling, and
remains inside AD for every prediction. No prediction-dependent trace masks.

SPDX-License-Identifier: MIT
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable
import torch.nn.functional as F

VARIANT = 'soft_dtw_divergence_float64_fixed_antialias_trace_mean_v1'
PINNED_PYSDTW_VERSION = '0.0.5'


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name+' must be finite and strictly positive')
    return value


def _pair(x: Tensor, y: Tensor):
    if not isinstance(x, Tensor) or not isinstance(y, Tensor):
        raise TypeError('Data must be torch tensors')
    if x.dtype != torch.float64 or y.dtype != torch.float64:
        raise TypeError('This qualified objective requires float64 data')
    if x.device != y.device or x.shape != y.shape or x.ndim < 1:
        raise ValueError('Require identical [...,time] shapes and devices')
    if x.numel()==0 or x.shape[-1] < 2 or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError('Require finite traces with at least two time samples')


def _kernels():
    try:
        import pysdtw
        from pysdtw import sdtw_cpu, sdtw_cuda
    except ImportError as error:
        raise ImportError('Fast soft-DTW requires pysdtw==0.0.5 and compatible numba/llvmlite') from error
    if pysdtw.__version__ != PINNED_PYSDTW_VERSION:
        raise RuntimeError('Qualified kernel dependency must be pysdtw==0.0.5')
    return sdtw_cpu, sdtw_cuda


def backend_provenance():
    """Call outside timers; record the exact kernel implementation used."""
    cpu, gpu = _kernels()
    return dict(variant=VARIANT, pysdtw_version=PINNED_PYSDTW_VERSION,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_sources_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                               for m in (cpu,gpu)},
        fast_backend_derivative_order=1, ground_cost='exact squared difference',
        gamma_precision='Python float64, never CUDA FloatTensor',
        state_precision='D/R/E float64 throughout')


class _Float64SoftDTW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, distance, gamma, bandwidth):
        cpu, gpu = _kernels()
        if distance.dtype != torch.float64 or distance.ndim != 3:
            raise TypeError('Expected float64 [trace,N,M] ground-cost matrices')
        distance = distance.contiguous()
        batch,n,m = distance.shape
        lengths = torch.tensor([n,m],device=distance.device,dtype=torch.int64).expand(batch,2).contiguous()
        ctx.gamma,ctx.bandwidth = float(gamma),float(bandwidth)
        if distance.is_cuda:
            from numba import cuda
            stream = cuda.external_stream(torch.cuda.current_stream(distance.device).cuda_stream)
            r = torch.full((batch,n+2,m+2),math.inf,dtype=distance.dtype,device=distance.device)
            r[:,0,0] = 0.
            threads=min(max(n,m),gpu.MAX_THREADS_PER_BLOCK)
            passes=max(n,m)//gpu.MAX_THREADS_PER_BLOCK+1
            gpu.compute_softdtw_cuda[batch,threads,stream](
                cuda.as_cuda_array(distance.detach()),ctx.gamma,ctx.bandwidth,
                cuda.as_cuda_array(lengths),passes,n+m-1,cuda.as_cuda_array(r))
        else:
            r_numpy=cpu.compute_softdtw(distance.detach().numpy(),lengths.numpy(),ctx.gamma,ctx.bandwidth)
            r=torch.from_numpy(r_numpy).to(dtype=distance.dtype,device=distance.device)
        ctx.save_for_backward(distance,r,lengths)
        return r[:,n,m].clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        cpu,gpu=_kernels();distance,r,lengths=ctx.saved_tensors
        batch,n,m=distance.shape
        if distance.is_cuda:
            from numba import cuda
            stream=cuda.external_stream(torch.cuda.current_stream(distance.device).cuda_stream)
            padded=F.pad(distance.detach(),(1,1,1,1))
            work=r.clone();e=torch.zeros_like(work)
            work[:,:,-1]=-math.inf;work[:,-1,:]=-math.inf
            work[:,-1,-1]=work[:,-2,-2];e[:,-1,-1]=1.
            threads=min(max(n,m),gpu.MAX_THREADS_PER_BLOCK)
            passes=max(n,m)//gpu.MAX_THREADS_PER_BLOCK+1
            gpu.compute_softdtw_backward_cuda[batch,threads,stream](
                cuda.as_cuda_array(padded),cuda.as_cuda_array(work),1./ctx.gamma,ctx.bandwidth,
                cuda.as_cuda_array(lengths),passes,n+m-1,cuda.as_cuda_array(e))
            derivative=e[:,1:-1,1:-1]
        else:
            e=cpu.compute_softdtw_backward(distance.detach().numpy(),r.detach().numpy().copy(),
                                          lengths.numpy(),ctx.gamma,ctx.bandwidth)
            derivative=torch.from_numpy(e).to(dtype=distance.dtype,device=distance.device)
        return derivative*grad_output[:,None,None],None,None


def soft_dtw_reference(x: Tensor,y: Tensor,gamma=.1,bandwidth=None):
    """Small pure-Torch, fully differentiable scalar-trace DP reference.

    Inputs are [trace,time]. This independent check is intentionally not a
    production long-record implementation and is limited to128 samples.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError('Reference inputs must be [trace,time] with equal trace count')
    if max(x.shape[-1],y.shape[-1])>128:
        raise ValueError('Pure-Torch reference is limited to128 samples')
    gamma=_positive(gamma,'gamma');batch,n=x.shape;m=y.shape[1]
    inf=torch.full((batch,),math.inf,dtype=x.dtype,device=x.device)
    previous=[torch.zeros_like(inf)]+[inf]*(m)
    for i in range(1,n+1):
        row=[inf]
        for j in range(1,m+1):
            if bandwidth and abs(i-j)>bandwidth:
                row.append(inf);continue
            candidates=torch.stack([previous[j-1],previous[j],row[-1]],dim=-1)
            row.append((x[:,i-1]-y[:,j-1]).square()-gamma*torch.logsumexp(-candidates/gamma,dim=-1))
        previous=row
    return previous[-1]


def soft_dtw_cost(x: Tensor,y: Tensor,gamma=.1,*,backend='pysdtw',bandwidth=None):
    """One unnormalized soft-DTW discrepancy for each [trace,time] pair."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError('Expected [trace,time] data with equal trace counts')
    if x.dtype != torch.float64 or y.dtype != torch.float64 or x.device != y.device:
        raise TypeError('Soft-DTW data must be float64 on the same device')
    gamma=_positive(gamma,'gamma')
    band=0 if bandwidth is None else int(bandwidth)
    if band<0 or (bandwidth is not None and float(bandwidth)!=band):
        raise ValueError('bandwidth must be a nonnegative integer number of processed samples')
    if backend=='reference':return soft_dtw_reference(x,y,gamma,band)
    if backend!='pysdtw':raise ValueError('backend must be pysdtw or reference')
    distance=(x[:,:,None]-y[:,None,:]).square()
    return _Float64SoftDTW.apply(distance,gamma,float(band))


class PreparedSoftDTWDivergence(nn.Module):
    """One-shot divergence with fixed observation preprocessing/self cost.

    All traces are included, including zero observed traces. Neither predicted
    values nor gradients choose a mask. Loss is mean over traces of the
    unmodified divergence divided by processed_Nt. No positivity clamp.
    Fixed observed preparation may be cached, identically for compared methods.
    """
    def __init__(self,observed: Tensor,dt: float,*,gamma=.1,scale=1.,decimation=20,
                 anti_alias_cutoff_hz=20.,anti_alias_transition_hz=5.,
                 backend='pysdtw',bandwidth: Optional[int]=None):
        super().__init__();_pair(observed,observed)
        if observed.requires_grad:raise ValueError('Observed data and its self cost must be fixed')
        self.dt=_positive(dt,'dt');self.gamma=_positive(gamma,'gamma')
        if isinstance(decimation,bool) or int(decimation)!=decimation or decimation<1:
            raise ValueError('decimation must be a positive integer')
        self.decimation=int(decimation);self.backend=backend;self.bandwidth=bandwidth
        self.data_shape=tuple(observed.shape);self.nt=observed.shape[-1]
        self.trace_count=observed.numel()//self.nt
        self.processed_nt=(self.nt-1)//self.decimation+1
        self.cutoff=None if anti_alias_cutoff_hz is None else _positive(anti_alias_cutoff_hz,'anti_alias_cutoff_hz')
        self.transition=_positive(anti_alias_transition_hz,'anti_alias_transition_hz')
        if self.decimation>1 and self.cutoff is None:
            raise ValueError('Decimation requires an explicit anti-alias filter')
        if self.cutoff is not None:
            if self.cutoff<=self.transition:
                raise ValueError('Anti-alias cutoff must exceed its positive transition')
            if self.cutoff+self.transition>1./(2*self.dt*self.decimation):
                raise ValueError('Antialias response must be zero by the processed Nyquist frequency')
        scale=torch.as_tensor(scale,dtype=observed.dtype,device=observed.device)
        if scale.requires_grad or not bool(torch.isfinite(scale).all()) or not bool((scale>0).all()):
            raise ValueError('Normalization scale must be fixed, positive and finite')
        if tuple(scale.shape)==tuple(observed.shape[:-1]) and scale.ndim:
            scale=scale[...,None]
        try:torch.broadcast_shapes(observed.shape,scale.shape)
        except RuntimeError as error:raise ValueError('Scale cannot broadcast to observed data') from error
        self.register_buffer('scale',scale.detach().clone())
        self.register_buffer('observed',observed.detach().clone())
        if self.cutoff is not None:
            frequencies=torch.fft.rfftfreq(3*self.nt,self.dt,device=observed.device,dtype=observed.dtype)
            position=((frequencies-(self.cutoff-self.transition))/(2*self.transition)).clamp(0,1)
            response=.5+.5*torch.cos(torch.pi*position)
        else:response=torch.empty(0,dtype=observed.dtype,device=observed.device)
        self.register_buffer('response',response)
        with torch.no_grad():
            processed=self.preprocess(observed)
            self_cost=soft_dtw_cost(processed,processed,self.gamma,backend=self.backend,bandwidth=self.bandwidth)
        self.register_buffer('processed_observed',processed)
        self.register_buffer('observed_self_cost',self_cost)

    def preprocess(self,data: Tensor):
        if data.shape!=self.data_shape:raise ValueError('Data shape differs from prepared observation')
        normalized=data/self.scale
        if not bool(torch.isfinite(normalized).all()):raise ValueError('Normalized data must remain finite')
        if self.cutoff is not None:
            padded=F.pad(normalized,(self.nt,self.nt))
            filtered=torch.fft.irfft(torch.fft.rfft(padded,dim=-1)*self.response,n=3*self.nt,dim=-1)
            normalized=filtered[...,self.nt:2*self.nt]
        return normalized[...,::self.decimation].reshape(self.trace_count,self.processed_nt)

    def trace_losses(self,predicted: Tensor):
        _pair(predicted,self.observed);p=self.preprocess(predicted);o=self.processed_observed
        cross=soft_dtw_cost(p,o,self.gamma,backend=self.backend,bandwidth=self.bandwidth)
        predicted_self=soft_dtw_cost(p,p,self.gamma,backend=self.backend,bandwidth=self.bandwidth)
        return (cross-.5*predicted_self-.5*self.observed_self_cost)/self.processed_nt

    def forward(self,predicted: Tensor):return self.trace_losses(predicted).mean()

    def configuration(self):
        return dict(variant=VARIANT,gamma=self.gamma,backend=self.backend,
            original_nt=self.nt,original_dt_s=self.dt,decimation=self.decimation,
            processed_nt=self.processed_nt,processed_dt_s=self.dt*self.decimation,
            anti_alias_cutoff_hz=self.cutoff,anti_alias_transition_hz=self.transition,
            bandwidth_processed_samples=self.bandwidth,trace_count=self.trace_count,
            normalization='Fixed input scale; each trace divergence / processed_Nt; one trace mean per shot',
            ground_cost='exact squared normalized amplitude difference',
            observed_self_cost_cached=True,prediction_self_both_arguments_differentiated=True,
            prediction_mask=False,negative_divergence_clamp=False)


def soft_dtw_divergence(predicted: Tensor,observed: Tensor,dt: float,**config):
    """Stateless one-shot entry; use PreparedSoftDTWDivergence for equal caching."""
    return PreparedSoftDTWDivergence(observed,dt,**config)(predicted)
