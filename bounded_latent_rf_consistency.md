# Bounded-Latent Rectified-Flow Consistency Models

## 1. Setup

We consider data samples

\[
x \sim p_{\text{data}}.
\]

A deterministic autoencoder maps data to a bounded latent space:

\[
z_1 = E_\psi(x) \in [-1,1]^d,
\]

where the encoder uses a final `tanh` activation. The decoder maps latent samples back to data space:

\[
\hat x = D_\psi(z).
\]

The base noise distribution is chosen to have the same compact support as the data latents:

\[
z_0 \sim \mathcal U([-1,1]^d).
\]

This gives a simple bounded latent transport problem:

\[
z_0 \sim p_0, \qquad z_1 \sim p_1,
\]

where both distributions live inside the same hypercube.

---

## 2. Rectified-flow path

We use the standard rectified-flow interpolation:

\[
z_t = (1-t)z_0 + t z_1,
\qquad t \in [0,1].
\]

The convention is:

\[
t=0: \text{noise}, \qquad t=1: \text{data latent}.
\]

For a paired sample \((z_0,z_1)\), the straight-line velocity is

\[
\frac{d z_t}{dt} = z_1 - z_0.
\]

A rectified-flow teacher is trained to approximate the Eulerian conditional velocity field:

\[
v_\phi(z_t,t) \approx \mathbb E[z_1-z_0 \mid z_t, t].
\]

The RF teacher objective is:

\[
\mathcal L_{\text{RF}}
=
\mathbb E_{z_0,z_1,t}
\left[
\left\|
v_\phi(z_t,t) - (z_1-z_0)
\right\|^2
\right].
\]

---

## 3. Consistency student as a flow map

The consistency student is not a velocity field. It is a learned flow map:

\[
C_\theta(z,t) \approx \Phi^{v_\phi}_{t\to 1}(z),
\]

where \(\Phi^{v_\phi}_{t\to 1}\) denotes the endpoint obtained by following the RF teacher ODE from time \(t\) to time \(1\):

\[
\frac{d z}{dt}=v_\phi(z,t).
\]

Thus, \(C_\theta(z,t)\) should directly predict the final data latent.

The desired boundary condition is:

\[
C_\theta(z,1)=z.
\]

A simple boundary-preserving parameterization is:

\[
C_\theta(z,t)=t z + (1-t)h_\theta(z,t).
\]

In the bounded-latent version, one can optionally use:

\[
h_\theta(z,t)=\tanh \tilde h_\theta(z,t),
\]

so that outputs remain bounded. However, for JVP-based training, an unbounded \(h_\theta\) can sometimes be better conditioned.

---

## 4. Continuous-time consistency condition

Along a teacher RF trajectory,

\[
\frac{d z(t)}{dt}=v_\phi(z(t),t).
\]

A perfect consistency map should be constant along this trajectory:

\[
C_\theta(z(t),t)=z_1.
\]

Therefore,

\[
\frac{d}{dt}C_\theta(z(t),t)=0.
\]

By the chain rule,

\[
\frac{d}{dt}C_\theta(z(t),t)
=
\partial_t C_\theta(z,t)
+
J_z C_\theta(z,t)v_\phi(z,t).
\]

So the continuous-time consistency objective is:

\[
\boxed{
\mathcal L_{\text{RF-sCM}}
=
\mathbb E_{z_0,z_1,t}
\left[
\left\|
\partial_t C_\theta(z_t,t)
+
J_z C_\theta(z_t,t)v_\phi(z_t,t)
\right\|^2
\right].
}
\]

This is the JVP-only sCM-style distillation objective.

---

## 5. JVP implementation

The residual can be computed using a Jacobian-vector product:

\[
\operatorname{JVP}_{(z,t)} C_\theta
\left[
(v_\phi(z,t), 1)
\right].
\]

In PyTorch:

```python
_, tangent = jvp(
    C_fn,
    primals=(zt, t),
    tangents=(v, torch.ones_like(t)),
)
```

where:

```python
def C_fn(z_in, t_in):
    return student(z_in, t_in)
```

The loss is:

```python
residual = tangent / norm
loss_raw = residual.pow(2).mean()
loss = (jvp_dt_ref ** 2) * loss_raw
```

The normalization

\[
\text{norm}=\sqrt{1+\frac{1}{d}\|v\|^2}
\]

is useful because the tangent vector contains both a latent-space component and a time component.

The factor

\[
\Delta t_{\text{ref}}^2
\]

is a scale conversion. The raw JVP loss has derivative units, while endpoint MSE has value units. Since

\[
C(z_{t+\Delta t},t+\Delta t)-C(z_t,t)
\approx
\Delta t
\frac{d}{dt}C(z_t,t),
\]

the squared finite-step consistency error is approximately

\[
\Delta t^2
\left\|
\frac{d}{dt}C(z_t,t)
\right\|^2.
\]

Thus,

\[
\mathcal L_{\text{scaled JVP}}
=
\Delta t_{\text{ref}}^2
\mathcal L_{\text{raw JVP}}.
\]

---

## 6. JVP-only RF-sCM distillation pseudocode

```python
# Given:
#   autoencoder ae
#   trained RF teacher v_phi
#   consistency student C_theta

for step in range(num_steps):

    # sample data latent
    x = sample_data()
    z1 = ae.encode(x)                         # z1 in [-1,1]^d

    # sample uniform latent noise
    z0 = uniform(-1, 1, shape=z1.shape)

    # sample time
    t = uniform(0, 1)
    t = clamp(t, eps, 1 - eps)

    # interpolate
    zt = (1 - t) * z0 + t * z1

    # teacher velocity
    with no_grad:
        v = teacher(zt, t)

    # JVP of consistency map along teacher trajectory
    tangent = JVP(
        C_theta,
        inputs=(zt, t),
        tangent=(v, 1)
    )

    # tangent normalization
    norm = sqrt(1 + mean(v ** 2, dim=-1, keepdim=True))

    # scaled JVP loss
    loss = jvp_dt_ref ** 2 * mean((tangent / norm) ** 2)

    update student
```

This trains the student to solve the flow-map PDE induced by the RF teacher.

---

## 7. Teacher-free Consistency Training analogue

The distillation version uses the trained teacher velocity:

\[
v_\phi(z_t,t).
\]

A teacher-free analogue replaces this with the known pairwise RF velocity:

\[
v_{\text{pair}} = z_1-z_0.
\]

The objective becomes:

\[
\boxed{
\mathcal L_{\text{RF-CT}}
=
\mathbb E_{z_0,z_1,t}
\left[
\left\|
\partial_t C_\theta(z_t,t)
+
J_z C_\theta(z_t,t)(z_1-z_0)
\right\|^2
\right].
}
\]

This is the direct RF analogue of consistency training from scratch.

The key difference is:

```python
# RF-sCM distillation
v = teacher(zt, t)

# RF-CT from scratch
v = z1 - z0
```

---

## 8. RF-CT pseudocode

```python
# Given:
#   autoencoder ae
#   consistency student C_theta
# No teacher is needed.

for step in range(num_steps):

    # sample data latent
    x = sample_data()
    z1 = ae.encode(x)

    # sample uniform noise
    z0 = uniform(-1, 1, shape=z1.shape)

    # sample time
    t = uniform(0, 1)
    t = clamp(t, eps, 1 - eps)

    # interpolate
    zt = (1 - t) * z0 + t * z1

    # pairwise straight-line velocity
    v = z1 - z0

    # JVP of student along the straight path
    tangent = JVP(
        C_theta,
        inputs=(zt, t),
        tangent=(v, 1)
    )

    # normalize and scale
    norm = sqrt(1 + mean(v ** 2, dim=-1, keepdim=True))
    loss = jvp_dt_ref ** 2 * mean((tangent / norm) ** 2)

    update student
```

This removes both teacher training and teacher ODE integration.

---

## 9. Distillation vs CT interpretation

### RF-sCM distillation

Uses the learned Eulerian RF field:

\[
v_\phi(z,t)\approx \mathbb E[z_1-z_0 \mid z_t=z].
\]

It trains \(C_\theta\) to be constant along the teacher ODE characteristics.

Pros:

- Lower-variance tangent field.
- More closely follows the learned RF transport.
- Natural analogue of sCM/CD.

Cons:

- Requires training a teacher velocity model.
- Student quality is limited by teacher quality.

### RF-CT

Uses the pairwise straight-line velocity:

\[
z_1-z_0.
\]

It trains \(C_\theta\) to be constant along sampled straight paths from noise to data.

Pros:

- No teacher.
- No ODE integration.
- Very simple in the bounded latent setup.
- Directly uses the known RF interpolation.

Cons:

- Higher-variance tangent target.
- Different random pairings can induce inconsistent directions in the same region.
- May require more tuning or larger batches.

---

## 10. Sampling

### One-step sampling

Sample uniform noise:

\[
z_0\sim\mathcal U([-1,1]^d).
\]

Predict the endpoint directly:

\[
\hat z_1 = C_\theta(z_0,0).
\]

Decode:

\[
\hat x = D_\psi(\hat z_1).
\]

Pseudocode:

```python
z0 = uniform(-1, 1, shape=(n, latent_dim))
t0 = zeros(n, 1)
z1_hat = student(z0, t0)
x_hat = ae.decode(z1_hat)
```

---

## 11. Multi-step consistency sampling

For \(k\)-step consistency sampling, choose times

\[
0=t_0<t_1<\dots<t_{k-1}<1.
\]

At each step, predict the endpoint, then re-interpolate between the original noise \(z_0\) and the current endpoint prediction:

\[
\hat z_1^{(i)} = C_\theta(z_{t_i},t_i),
\]

\[
z_{t_{i+1}}
=
(1-t_{i+1})z_0
+
t_{i+1}\hat z_1^{(i)}.
\]

After the final step, decode:

\[
\hat x = D_\psi(\hat z_1^{(k-1)}).
\]

Pseudocode:

```python
z0 = uniform(-1, 1, shape=(n, latent_dim))

times = linspace(0, 1, k + 1)[:-1]

zt = z0

for i, t in enumerate(times):

    z1_hat = student(zt, t)

    if i + 1 < k:
        t_next = times[i + 1]
        zt = (1 - t_next) * z0 + t_next * z1_hat

x_hat = ae.decode(z1_hat)
```

This is still endpoint-consistency sampling, not arbitrary-time flow-map sampling.

---

## 12. Relation to velocity prediction

The RF teacher is a velocity model:

\[
v_\phi(z,t).
\]

The consistency student is not a velocity model. It is an amortized solver:

\[
C_\theta(z,t)\approx \Phi_{t\to1}(z).
\]

A velocity student would learn:

\[
u_\theta(z,t)\approx v_\phi(z,t),
\]

but sampling would still require numerical integration.

The consistency student instead learns the integrated endpoint directly and therefore supports one-step generation.

A useful alternative parameterization is an average-velocity endpoint map:

\[
C_\theta(z,t)
=
z+(1-t)a_\theta(z,t),
\]

where

\[
a_\theta(z,t)
\approx
\frac{\Phi_{t\to1}(z)-z}{1-t}.
\]

This is not the instantaneous RF velocity. It is the average velocity needed to jump from time \(t\) to the endpoint.

---

## 13. Relation to AYF-style flow maps

The current method learns an endpoint-only flow map:

\[
C_\theta(z,t)=F_\theta(z,t,1).
\]

An arbitrary-time flow-map model would learn:

\[
F_\theta(z,s,r)\approx \Phi_{s\to r}(z).
\]

The endpoint-only method is the special case:

\[
r=1.
\]

An AYF-style sampler would step through intermediate states:

\[
z_{t_{i+1}} = F_\theta(z_{t_i},t_i,t_{i+1}),
\]

rather than repeatedly predicting the endpoint and re-interpolating.

Thus:

```text
Current method:
    endpoint consistency map C(z,t)

AYF-style extension:
    arbitrary-time flow map F(z,s,r)
```

---

## 14. Why bounded uniform latents simplify the method

The bounded-latent setup gives several simplifications.

### Same support for noise and data

\[
z_0\sim\mathcal U([-1,1]^d),
\qquad
z_1=E(x)\in[-1,1]^d.
\]

Both endpoints live in the same compact domain.

### Straight path stays inside the domain

Because the cube is convex,

\[
z_t=(1-t)z_0+tz_1 \in [-1,1]^d.
\]

The model never has to handle arbitrarily large Gaussian-noise latents.

### No EDM-style noise schedule

There is no need for:

- \(\sigma_{\min}\)
- \(\sigma_{\max}\)
- \(\sigma_{\text{data}}\)
- log-SNR
- EDM preconditioning

The path is directly parameterized by \(t\in[0,1]\).

### Simple boundary condition

The endpoint condition is simply:

\[
C_\theta(z,1)=z.
\]

This can be enforced directly by parameterization.

### Direct tangent field

For RF-sCM, the tangent is just the teacher velocity:

\[
(v_\phi(z,t), 1).
\]

For RF-CT, the tangent is just the pair velocity:

\[
(z_1-z_0, 1).
\]

---

## 15. Practical implementation notes

### Time embeddings

High-frequency Fourier time embeddings can destabilize JVP training because the loss differentiates through time. If the time embedding contains

\[
\sin(2\pi 2^k t),
\]

then

\[
\frac{d}{dt}\sin(2\pi 2^k t)
=
2\pi 2^k \cos(2\pi 2^k t),
\]

which can produce huge artificial \(\partial_t C_\theta\) terms.

Prefer:

- low-frequency Fourier embeddings
- polynomial time features
- small MLP time embeddings

### LayerNorm

LayerNorm can hurt JVP training because the objective differentiates the student with respect to its inputs. Normalization layers modify the input Jacobian and can make

\[
J_z C_\theta(z,t)v(z,t)
\]

ill-conditioned.

In practice, plain MLPs without LayerNorm worked better.

### JVP scaling

The raw JVP loss can be much larger than endpoint MSE because it has derivative units. Use:

\[
\mathcal L = \Delta t_{\text{ref}}^2 \mathcal L_{\text{raw JVP}}.
\]

Typical values:

```python
jvp_dt_ref = 0.02
jvp_dt_ref = 0.05
```

If training is stable, larger values can be tried.

### Endpoint anchors

Although the latest method can work with pure JVP, an optional sparse endpoint anchor can stabilize training:

\[
\mathcal L
=
\mathcal L_{\text{JVP}}
+
\lambda \mathcal L_{\text{endpoint}}.
\]

For distillation:

\[
\mathcal L_{\text{endpoint}}
=
\left\|
C_\theta(z_t,t)
-
\Phi_{t\to1}^{v_\phi}(z_t)
\right\|^2.
\]

For teacher-free CT:

\[
\mathcal L_{\text{endpoint}}
=
\left\|
C_\theta(z_t,t)-z_1
\right\|^2.
\]

However, the focus of the latest method is pure JVP training.

---

## 16. Summary of methods

### RF-sCM, teacher-distilled JVP

Train an RF teacher:

\[
v_\phi(z,t).
\]

Then train the consistency map with:

\[
\mathcal L_{\text{RF-sCM}}
=
\mathbb E
\left[
\left\|
\partial_t C_\theta
+
J_z C_\theta v_\phi
\right\|^2
\right].
\]

This is the continuous-time consistency distillation version.

### RF-CT, teacher-free pairwise JVP

Skip the teacher and use the known pairwise velocity:

\[
v=z_1-z_0.
\]

Train with:

\[
\mathcal L_{\text{RF-CT}}
=
\mathbb E
\left[
\left\|
\partial_t C_\theta
+
J_z C_\theta (z_1-z_0)
\right\|^2
\right].
\]

This is the teacher-free consistency training analogue.

---

## 17. Core claim

The method studies consistency/flow-map learning in bounded autoencoder latent spaces where the base noise and data latents share the same compact support.

The key simplification is:

\[
z_0\sim\mathcal U([-1,1]^d),
\qquad
z_1=E(x)\in[-1,1]^d.
\]

This makes rectified-flow consistency especially simple:

\[
z_t=(1-t)z_0+tz_1,
\]

\[
C_\theta(z,1)=z,
\]

\[
\mathcal L
=
\left\|
\operatorname{JVP}_{(z,t)} C_\theta[(v,1)]
\right\|^2.
\]

The teacher-distilled version uses:

\[
v=v_\phi(z,t),
\]

while the consistency-training version uses:

\[
v=z_1-z_0.
\]
