# Optional shared data-domain soft-DTW fixture

For the fast40 profile and default synthetic3D acceptance,
[scripts/soft_dtw_objective.py](../scripts/soft_dtw_objective.py)
is a self-contained data-only implementation copied byte-for-byte from the
qualified shared receiver objective. Its SHA256 is
`f37dcaa66d9399b27545563fe125672e95a83535e4b7ba1ae51e5a60d027fefc`;
[the included MIT notice](../LICENSE.soft-dtw-objective) retains its copyright.
It contains no propagator, native adjoint, solver adapter, parameter map or
optimizer. Reusing this generic objective is not inheriting a target adapter.

The module provides `PreparedSoftDTWDivergence`, `soft_dtw_divergence` and the
small independent `soft_dtw_reference`. Inputs have shape `[trace,time]` or
other leading trace axes and must be float64. A complete shot remains one
logical objective; the caller applies the single mean over all declared shots.
The implementation differentiates both copies in the predicted self term,
keeps the observed self cost fixed, and includes the zero-padded anti-alias
filter and decimation in prediction AD. It does not choose native source units,
receiver components, time alignment, observation scales or stage filtering.

## Backends and resources

- `backend='reference'` needs PyTorch only and is suitable for independent small
  value/VJP checks. Its scalar dynamic-program graph is not a production
  recommendation for313-sample traces across200 receivers and40 shots.
- `backend='pysdtw'` uses external MIT-licensed `pysdtw==0.0.5` CPU/CUDA DP kernels
  with the bundled precision-preserving wrapper. The fast path is first-order
  only. Record `backend_provenance()` before the timer and the actual resource
  results on the target data. Dependencies are external, not bundled kernels.
- The previously qualified optional combination was NumPy2.0.x,
  NumBa0.60.0/llvmlite0.43.0 and PySDTW0.0.5. Native solver environments may need
  different dependency isolation; an AD-side environment or process boundary
  can preserve native execution. Do not silently change propagation libraries
  or install conflicting packages into a working native environment.

A setup command for an isolated compatible AD environment is:

```bash
python -m pip install 'pysdtw==0.0.5' 'numba==0.60.0' \
  'llvmlite==0.43.0' 'numpy>=1.22,<2.1'
```

## Apply the profile outside the native time loop

The target adapter must still implement and independently verify this boundary:

1. Bind native sample/component conventions and map observation and prediction
   to the declared common time axis. Keep the prediction map in AD.
2. Apply the profile's stage low-pass filter. Compute one fixed positive observed
   RMS per shot/stage over the declared trace/component/time samples.
3. Construct `PreparedSoftDTWDivergence` from that stage-filtered observation,
   with `dt=.003`, `gamma=.1`, the fixed RMS as `scale`, `decimation=8`,
   `anti_alias_cutoff_hz=10.`, `anti_alias_transition_hz=2.` and no alignment band.
4. Apply the same stage filtering to the current predicted data, evaluate the
   module, then call `torch.autograd.grad` with respect to the original native
   receiver-data leaf. That VJP includes the sampling and filtering transpose.
5. Convert the resulting cotangent to the target's native source interface and
   verify the complete model derivative. The module does not perform that step.

The numbers above belong to the planar2D fast40 profile. For3D, export the
[fixed salt case](salt3d-acceptance.md). Its declared reference grid is
1051samples at0.01s; decimation5 with cutoff8Hz and transition2Hz gives211samples
at0.05s, with the stopband zero at the10Hz processed Nyquist. Bind the actual
native time grid/components to that declared observation map inside AD; verify
the mapping and full model derivative. Do not copy the planar2D numbers or
class constructor defaults, or select another3D model to make the fixture pass.

All declared shot contributions use the same current model; streaming their separate
AD graphs can bound DP memory without changing the full-shot mean or permitting
an update between shots. Cache the observed preparation only for unchanged
observation, filter, scale, gamma and cost settings.

Run the optional fixture checks before the target's actual derivative checks:

```bash
python scripts/check_soft_dtw_fixture.py --backend reference --output fixture.json
# With the optional dependencies installed:
python scripts/check_soft_dtw_fixture.py --backend pysdtw --output fast_fixture.json
```

These small CPU checks cover data-domain value/VJP behavior only. A historical
qualification or a passed fixture check does not validate a new native
resampling/source conversion, full40-shot gradient, long-record resources,
inversion outcome or convergence. Those checks remain application work.
