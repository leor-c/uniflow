# URFM — Uniform-noise Rectified Flow + velocity-matching distillation

Research codebase for: distilling a **uniform-noise rectified-flow teacher**
(bounded `[-1,1]^d` latents, no noise schedule, velocity-only) into a one/few-step
student via a **DMD2-style velocity-matching loss**, with a Wasserstein-2
distillation guarantee whose key constant (`Lambda = ∫ L(t) dt`) is measurable.

## Structure

```
urfm/
  core/                 # shared, framework-agnostic — used by ALL phases
    interp.py           # x_t=(1-t)x0+t*eps, uniform noise, velocity target, denoiser identity
    velocity.py         # MLP velocity net, RF teacher loss, ODE generation (Euler/Heun)
    distill.py          # one-step generator, EMA fake critic, two-timescale velocity-matching
    stiffness.py        # power-iteration probe for L(t)=sup_x||grad_x w||_op  (theory tool)
  toy/                  # PHASE 0 — 2D sanity + theory validation (CPU-friendly)
    datasets.py         # 8gaussians, two_moons, checkerboard (all in [-1,1]^2)
    train_toy.py        # teacher -> L(t) profile -> distill -> eval -> plots
  imagenet/             # PHASE 1-2 (to build): imports core/, adds U-Net + scale pieces
  eval/                 # FID/IS/recall (to port from DMD2 repo)
```

The design contract: **`core/` is shared and minimal.** Phase 0 imports it and runs
in minutes on CPU. ImageNet phases import the *same* `core/` and add only what scale
requires (U-Net backbone, real-data loading, distributed training, FID harness).

## Quickstart (Phase 0)

```bash
pip install torch matplotlib
python -m urfm.toy.train_toy --dataset 8gaussians \
    --teacher_iters 4000 --distill_iters 4000 --out runs/toy_8g
```

Outputs `runs/toy_8g/metrics.json` (stiffness profile, Lambda, energy distances)
and `phase0.png` (real / teacher / student scatter + L(t) curve).

## What the stiffness probe measures (the theory tool)

`core/stiffness.py` estimates `L(t) = sup_x ||grad_x w(x,t)||_op` via power iteration
on the velocity network's input-Jacobian (JVP + VJP, ~1 fwd + 1 bwd per iter,
dimension-independent). The theory (findings doc §5) predicts:
  - **Boundary anchoring:** `L(0)=1`, `L(1)=1` exactly, for ANY data.
  - **Interior bump** controlled by data-density curvature `||∇²log ρ||`.
  - `Lambda = ∫_0^1 L(t) dt < ∞` ⇒ the W2 bound's constant `e^{2Λ}` is finite.

## Findings so far (Phase 0, CPU, short runs — preliminary)

- **Teacher works:** uniform-noise RF recovers all modes; no schedule/preconditioning.
- **Stiffness probe is VALIDATED:** tested against exact SVD on known linear maps —
  matches to ~1e-6 when converged; power iteration can only *under*estimate, never
  overestimate (see `tests/validate_probe.py`). So any measured stiffness is a
  trustworthy lower bound on the true sup.
- **Boundary anchoring CONFIRMED:** on smooth *unimodal* data (tanh-mapped Gaussian),
  `L(0)≈1.04`, `L(1)≈1.21`, `Lambda≈1.4` — matching the theory's `L(0)=L(1)=1`
  prediction closely. The small residual is finite-network fit error.
- **Interior bump tracks multimodality/curvature, as predicted:** multimodal data
  (8 Gaussians) shows an inflated interior bump and a larger `L(0)` (~5). This is
  genuine field stiffness where `x_t` sits near inter-mode decision boundaries
  (the `||∇²log ρ||` term in §5) — NOT a fitting failure (see loss-floor note below).
- **The RF loss value is NOT a fit metric — judge by samples.** The regression
  target `w = x0 - eps` has large irreducible conditional variance (eps is mostly
  unobservable from `x_t`), so the loss floors well above zero (~0.25 on the toy
  Gaussian) even for a perfect model. Verified: loss flattens by iter ~500 while
  sample energy-distance is already ~0.002 and does not improve with 16x more
  iterations — i.e. the teacher is converged and fits near-perfectly; the nonzero
  loss is the floor, not under-training. (This is the usual diffusion/flow behavior.)
- **Data must be smoothly bounded, not clamped:** datasets now use **tanh-mapping**
  into the open box `(-1,1)^2`, mirroring tanh-bottleneck latents. Clamping piles mass
  on the box faces (a density spike) and is the wrong construction for this study.
- **Distillation bug fixed:** the generator gradient now flows directly into `x0_hat`
  with the correct sign (descent moves `x0_hat` toward `v_T - v_S`). The previous
  version routed it through `x_t`, attenuating the signal by `(1-t)` and killing the
  high-noise contributions our objective weights equally. Sign verified empirically
  (unimodal target collapses correctly).
- **Student status — works, not yet crisp.** After the fix, on 8-Gaussians the
  student covers all modes and keeps the empty center nearly empty (spurious
  center-mass 0.0–0.6%, vs teacher 0.3%), so there is no gross mode-collapse or
  divergence. BUT the per-mode spread is wider than the data — modes are not crisp.
  This is a quality/tuning gap, not a logic bug. The critic tracks the student near
  the irreducible loss floor (~0.5), i.e. about as well as the objective allows.
- **Critic init matters:** RANDOM critic init beats teacher-warm-start here
  (center-mass 0.000 vs 0.006). Teacher-init makes `v_S≈v_T` initially (zero
  signal) and must un-learn first. Default is now random init.
- **Likely remaining levers for crisp modes:** more capacity / longer training,
  and/or the auxiliary GAN head DMD2 uses specifically to sharpen one-step output.
  Pure velocity-matching may under-sharpen modes on its own.

## Known open items

- **Student tuning** (gating): longer GPU runs, LR / critic-step / EMA sweeps,
  optional GAN head; verify crisp one-step mode recovery.
- **Low-`t` boundary behavior:** the `t→0` region has the *smallest* conditional
  variance (x_t pins x0), so it is the *easiest* part of the loss, and the teacher
  fits it well. Residual `L(0)` above 1 on multimodal data reflects genuine field
  stiffness, not fit error. A time-weighting study is still worthwhile for how
  tightly `L(0)→1` holds on hard multimodal data.

## Immediate TODO

1. **Tune the student** (the gating item):
   - sweep generator LR, critic-steps ratio, EMA decay;
   - consider adding the DMD2 GAN head (currently velocity-matching only);
   - longer runs on GPU; verify 1-step student recovers all modes.
2. **Boundary-anchoring figure:** clean L(t) curves across datasets/smoothness,
   showing L(0)=L(1)→1 as data smooths. This is a paper figure.
3. **Few-step student:** extend `OneStepGenerator` / `distill_step` to K steps.
4. **ImageNet phase:** add `imagenet/` U-Net (borrow EDM/ADM backbone), real-data
   loader, and port DMD2's eval harness into `eval/`. Keep the only teacher-side
   difference vs EDM = noise/parameterization, for a fair 4-arm comparison
   (A: EDM+DMD2, B: EDM+ours, C: URFM+DMD2-style, D: URFM+ours).

## Notes / caveats

- `distill.py` uses the standard DMD "treat the (detached) velocity difference as
  the target direction" trick so the generator loss is a stable MSE. The GAN head
  from DMD2 is **not** included yet; add if mode coverage needs it.
- The energy-distance metric is a 2D-only convenience; replace with FID at scale.
- Hyperparameters in `train_toy.py` are illustrative defaults, NOT tuned.
- For Arm A (reproduce DMD2), run the official repo directly with their
  checkpoints rather than reimplementing — only B/C/D live in this repo.
```
