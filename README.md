# UniFlow

Diffusion and flow models predominantly use Gaussian noise, which requires rescaling the latents to match its fixed unit scale.

This repo studies an alternative: uniform noise on [-1, 1] paired with tanh-bounded latents (a tanh applied to the encoder output). The supports coincide by construction, so there's nothing to rescale, and the bounded space can be enforced at generation time.

The study uses 1-Rectified Flow (RF) and compares the standard Gaussian noise baseline against the uniform noise variant, with an EDM baseline included for reference. It also evaluates Align Your Flow (AYF) distillation, training an AYF student for each baseline.

---

The repository compares three matched toy pipelines:

| Arm | Data latent | Base distribution | Learned models |
|---|---|---|---|
| Gaussian RF | Standardized unbounded encoder output | $\mathcal N(0,I)$ | RF teacher + AYF student |
| Uniform RF | $\tanh(E(x)) \in (-1,1)^d$ | $\mathcal U([-1,1]^d)$ | RF teacher + AYF student |
| EDM | Standardized unbounded encoder output | Gaussian perturbations | EDM teacher + AYF student |

This is intentionally a controlled 2D benchmark. The autoencoder and network
width/depth are matched across all arms. Gaussian RF and EDM train the same
unbounded autoencoder for reconstruction only, followed by a fixed post-training
latent rescaling: the latents are centred and divided by a *single global
scalar*, following latent-diffusion practice, so the transform is a similarity
and the latent geometry reaches the generative model undistorted. Uniform RF
adds the encoder's final `tanh` and a weak
moment penalty (weight $10^{-3}$) that targets the mean and second moment of
$\mathcal U([-1,1])$; its latents are not post-training standardized.

## Method

For both RF arms, independently sampled endpoints are joined by

$$
z_t = (1-t)z_0 + t z_1, \qquad t \sim \mathcal U(0,1),
$$

and the teacher regresses the conditional velocity with

$$
\mathcal L_{\mathrm{RF}}
= \mathbb E\left[\left\|v_\phi(z_t,t)-(z_1-z_0)\right\|^2\right].
$$

The AYF student is an arbitrary-interval flow map,

$$
C_\theta(z,t,s)=z+(s-t)f_\theta(z,t,s),
$$

trained to remain constant when its source moves along a teacher trajectory:

$$
\mathcal L_{\mathrm{AYF}}
= \mathbb E\left[\left\|
\partial_t C_\theta + J_z C_\theta\,v_\phi
\right\|^2\right].
$$

Following AYF, the tangent is normalised by its own norm,
$\mathrm{d}f/\mathrm{d}t \mapsto (\mathrm{d}f/\mathrm{d}t)/(\lVert
\mathrm{d}f/\mathrm{d}t\rVert + 0.1)$, which bounds the influence of the rare
large tangents that destabilise map distillation.

At inference, the same map supports one-step generation with $C(z_0,0,1)$ or
few-step generation by composing adjacent intervals.

The EDM teacher retains its standard preconditioned denoising objective with the
published Karras noise distribution and schedule. AYF distils diffusion teachers
inside the flow-matching framework rather than in $\sigma$ space, so the trained
denoiser is exposed as a flow-matching velocity field through the change of
variables $\sigma(t)=(1-t)/t$:

$$
v(z_t,t)=\frac{D\!\left(z_t/t,\ \sigma(t)\right)-z_t}{1-t}.
$$

The shared AYF student, objective, interval schedule and samplers then drive it
unchanged, so all three arms run the same algorithm with the same boundary
$C_\theta(z,t,t)=z$ and differ only in their teacher and base distribution. The
velocity stays $O(1)$ across the whole interval, unlike the preconditioned
unit-scale state $y=z/\sqrt{\sigma^2+\sigma_{\mathrm{data}}^2}$ used
previously, in which the true velocity collapses as
$O((\sigma_{\mathrm{data}}/\sigma)^2)$ at the noise end while the teacher's
denoiser error does not.

## Results

The default 8-Gaussians benchmark uses seeds 1–5, 32-dimensional latents, 20k
AE steps, 10k teacher steps, 10k student steps, and batch size 64. Each seed
generates 5,000 evaluation samples and evaluates 2,048 uniformly sampled points
with three complementary density diagnostics: generated-sample NLL under the
known target density (fidelity), held-out target NLL under a generated-sample
KDE (coverage), and the fraction beyond a three-noise-scale tube around the
clean modes or arcs. Results are the mean ± sample standard deviation across
seeds; lower is better. Figures use seed 1 as a
qualitative example and contain the target data, AE reconstruction, teacher ODE
samples, and one- and eight-step students.

The previous SWD, exact-W2, and MMD tables were removed because a single
symmetric average did not reflect the visible two-moons geometry. Run the
benchmark to regenerate the release tables with separate fidelity and coverage
values. Evaluation tensors are saved as ignored `.pt` files alongside the JSON
metrics so future metric changes do not require retraining.

### Qualitative samples

The following panels are seed 1 examples from that archived benchmark run.

**Gaussian RF + AYF**

![Gaussian RF and AYF samples](results/gaussian_rf_8gaussians-seed-1.png)

**Uniform RF + AYF (uniform noise + tanh latents)**

![Uniform RF and AYF samples](results/uniform_rf_8gaussians-seed-1.png)

**EDM**

![EDM teacher and consistency-student samples](results/edm_ayf_8gaussians-seed-1.png)

## Reproduce

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m toy.run_all
```

The default command reruns the five-seed benchmark while keeping seed 1 as the
only qualitative figure. Its new `edm_ayf_*` outputs are deliberately separate
from the archived `edm_*` consistency-student outputs shown above. When CUDA is
available, the seed runs execute in parallel spawned processes; each worker
runs the three pipelines for its seed sequentially. By default, one worker is
used per seed. Limit GPU memory use with `--workers N`, or pass `--workers 1`
for sequential execution. CPU runs remain sequential.

To continue an interrupted run, reuse completed seed files with:

```bash
python -m toy.run_all --resume
```

Raw per-seed metrics are written to `results/seeds/`; the aggregate, including
all individual values and the sample standard deviation, is written to
`results/summary_8gaussians.json` and a Markdown table is written to
`results/summary_8gaussians.md`. A corresponding grouped bar chart with
per-seed points and sample-standard-deviation error bars is written to
`results/summary_8gaussians.png`. `--resume` reuses completed seed files.

To generate qualitative figures for every seed, rerun with:

```bash
python -m toy.run_all --plot-all-seeds
```

This writes filenames such as `gaussian_rf_8gaussians-seed-3.png`. Because the
toy benchmark does not save model checkpoints, `--resume` cannot generate a
missing figure from an already completed run.

For a quick pipeline check:

```bash
python -m toy.run_all --seed 1 --ae_steps 10 --teacher_steps 10 --student_steps 10
```

Set one shared autoencoder latent dimension across all three benchmark arms with:

```bash
python -m toy.run_all --latent_dim 8
```

Individual arms can be run with `python -m toy.gaussian_rf`,
`python -m toy.uniform_rf`, or `python -m toy.edm_baseline`.
All generated figures and JSON metrics are written to `results/`.

The `mnist/` directory contains optional Uniform RF/AYF and EDM extensions.
They are not part of the reported comparison and require `torchvision` plus a
separate, substantially longer training run.

## Repository layout

```text
core/rf/       shared RF teacher and ODE utilities
core/ayf/      Gaussian-prior AYF objective and sampling
core/ayf_rf/   uniform-prior, bounded-latent AYF objective and sampling
core/edm/      EDM teacher, flow-matching wrapper, and consistency reference
toy/           primary 2D benchmark and runner
mnist/         optional image-space extension
results/       generated figures and metric summaries
```

## Scope

This repository contains research code, not a production library. The removed
ImageNet prototype was never validated and is deliberately excluded from the
public release. Large datasets, checkpoints, and local runs are ignored by Git.
