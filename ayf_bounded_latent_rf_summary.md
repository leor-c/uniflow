# AYF-Style Bounded-Latent Rectified-Flow Flow Maps

## 1. Motivation

The endpoint consistency model learns a map

\[
C_\theta(z_t,t)\to z_1,
\]

where \(z_1\) is the final data latent. This supports one-step generation, but multi-step sampling still repeatedly predicts the endpoint and re-interpolates with the original noise.

The Align Your Flow style generalization is to learn an arbitrary-interval flow map:

\[
C_\theta(z_t,t,s)\to z_s,
\]

where \(t\) is the source time and \(s\) is the target time.

This makes the model a general learned solver over intervals \([t,s]\), not only a learned solver to the endpoint \(s=1\).

---

## 2. Bounded-latent RF setup

We use a tanh-bounded autoencoder latent:

\[
z_1 = E_\psi(x)\in[-1,1]^d.
\]

The base distribution is uniform noise on the same compact domain:

\[
z_0\sim\mathcal U([-1,1]^d).
\]

The rectified-flow interpolation is

\[
z_t=(1-t)z_0+t z_1,
\qquad t\in[0,1].
\]

The convention is:

\[
t=0:\text{ noise},\qquad t=1:\text{ data latent}.
\]

The RF teacher learns the velocity field

\[
v_\phi(z_t,t)\approx \mathbb E[z_1-z_0\mid z_t,t].
\]

---

## 3. Endpoint consistency model

The endpoint model learns

\[
C_{\text{end}}(z,t)\approx \Phi^{v_\phi}_{t\to1}(z),
\]

where \(\Phi^{v_\phi}_{t\to1}\) is the RF teacher flow from \(t\) to \(1\).

A useful average-velocity parameterization is

\[
C_{\text{end}}(z,t)=z+(1-t)f_\theta(z,t).
\]

This enforces

\[
C_{\text{end}}(z,1)=z.
\]

---

## 4. AYF-style interval flow map

The generalized interval model is

\[
\boxed{
C_\theta(z,t,s)=z+(s-t)f_\theta(z,t,s)
}
\]

for \(0\le t\le s\le1\).

This enforces the identity boundary:

\[
C_\theta(z,t,t)=z.
\]

The model target is

\[
C_\theta(z,t,s)\approx \Phi^{v_\phi}_{t\to s}(z).
\]

The endpoint consistency model is the special case

\[
C_{\text{end}}(z,t)=C_\theta(z,t,1).
\]

The network output \(f_\theta\) can be interpreted as an average velocity over \([t,s]\):

\[
f_\theta(z,t,s)\approx
\frac{\Phi^{v_\phi}_{t\to s}(z)-z}{s-t}.
\]

It is not necessarily the instantaneous RF velocity.

---

## 5. EMD source-time JVP objective

For a true flow map,

\[
C_\theta(z_t,t,s)=z_s.
\]

If we move the source point \(z_t\) forward along the teacher trajectory while keeping the target time \(s\) fixed, the output should remain unchanged.

The teacher trajectory satisfies

\[
\frac{dz}{dt}=v_\phi(z,t).
\]

Therefore,

\[
\frac{d}{dt}C_\theta(z_t,t,s)=0.
\]

By the chain rule,

\[
\boxed{
\partial_t C_\theta(z,t,s)
+
J_z C_\theta(z,t,s)v_\phi(z,t)
=
0.
}
\]

The EMD loss is

\[
\boxed{
\mathcal L_{\text{EMD}}
=
\mathbb E
\left[
\left\|
\partial_t C_\theta(z_t,t,s)
+
J_z C_\theta(z_t,t,s)v_\phi(z_t,t)
\right\|^2
\right].
}
\]

This reduces to the endpoint RF-sCM objective when \(s=1\).

---

## 6. EMD JVP pseudocode

```python
z1 = ae.encode(x)
z0 = uniform(-1, 1, shape=z1.shape)

t, s = sample_time_pair()      # 0 <= t < s <= 1
zt = (1 - t) * z0 + t * z1

with no_grad:
    v = teacher(zt, t)

def C_source_fn(z_in, t_in):
    return student(z_in, t_in, s)

_, tangent = jvp(
    C_source_fn,
    primals=(zt, t),
    tangents=(v, ones_like(t)),
)

norm = sqrt(1 + mean(v ** 2, dim=-1, keepdim=True))
loss_raw = mean((tangent / norm) ** 2)
loss = jvp_dt_ref ** 2 * loss_raw
```

The factor \(jvp\_dt\_ref^2\) converts the derivative-scale JVP residual into the scale of a finite-difference consistency error.

---

## 7. LMD target-time objective

There is another natural flow-map condition. For fixed source \((z_t,t)\), as the target time \(s\) changes, the predicted target state should move according to the teacher velocity at the predicted state:

\[
\frac{\partial}{\partial s} C_\theta(z_t,t,s)
=
v_\phi(C_\theta(z_t,t,s),s).
\]

The LMD loss is

\[
\boxed{
\mathcal L_{\text{LMD}}
=
\mathbb E
\left[
\left\|
\partial_s C_\theta(z_t,t,s)
-
v_\phi(C_\theta(z_t,t,s),s)
\right\|^2
\right].
}
\]

Pseudocode:

```python
def C_target_fn(s_in):
    return student(zt, t, s_in)

z_s_hat, dC_ds = jvp(
    C_target_fn,
    primals=(s,),
    tangents=(ones_like(s),),
)

with no_grad:
    v_target = teacher(z_s_hat, s)

loss = mean((dC_ds - v_target) ** 2)
```

In practice, EMD is the closest continuation of the endpoint JVP method. LMD can be added with a small weight after EMD is stable.

---

## 8. Optional interval anchors

### Pairwise chord anchor

Since the RF path is explicitly defined by sampled pairs, a cheap interval target is

\[
z_s=(1-s)z_0+s z_1.
\]

The pairwise interval anchor is

\[
\mathcal L_{\text{pair}}
=
\|C_\theta(z_t,t,s)-z_s\|^2.
\]

This is not the same as teacher ODE distillation. It is mostly useful as a sanity check or weak anchor.

### Teacher interval anchor

A stronger but more expensive anchor uses the teacher ODE target:

\[
\mathcal L_{\text{teacher-anchor}}
=
\left\|
C_\theta(z_t,t,s)
-
\Phi^{v_\phi}_{t\to s}(z_t)
\right\|^2.
\]

This requires integrating the teacher for every batch.

---

## 9. Teacher-free AYF-CT analogue

The teacher-free version replaces the teacher velocity with the pairwise RF velocity:

\[
v_{\text{pair}}=z_1-z_0.
\]

The CT-style EMD loss becomes

\[
\boxed{
\mathcal L_{\text{AYF-CT}}
=
\mathbb E
\left[
\left\|
\partial_t C_\theta(z_t,t,s)
+
J_z C_\theta(z_t,t,s)(z_1-z_0)
\right\|^2
\right].
}
\]

This removes the teacher, but can be much harder with independent coupling because many sampled chords may cross, creating inconsistent directions for the same \((z_t,t)\).

---

## 10. AYF-style k-step sampling

The endpoint model performs multi-step sampling by repeatedly predicting \(z_1\) and re-interpolating.

The AYF-style interval flow map instead performs true intermediate stepping:

\[
z_{t_{i+1}}=C_\theta(z_{t_i},t_i,t_{i+1}).
\]

For a schedule

\[
0=t_0<t_1<\dots<t_k=1,
\]

sampling is:

```python
z = uniform(-1, 1, shape=(n, latent_dim))

for i in range(k):
    t = t_i
    s = t_{i+1}
    z = student(z, t, s)

x = ae.decode(z)
```

One-step generation is simply

\[
z_1 = C_\theta(z_0,0,1).
\]

---

## 11. Relationship to the endpoint method

Endpoint model:

\[
C_{\text{end}}(z,t)=z+(1-t)f_\theta(z,t).
\]

AYF interval model:

\[
C_\theta(z,t,s)=z+(s-t)f_\theta(z,t,s).
\]

Endpoint is the special case:

\[
C_{\text{end}}(z,t)=C_\theta(z,t,1).
\]

Expected behavior:

- Endpoint model may be strongest for one-step generation.
- Interval model should be better aligned with true multi-step generation.
- Interval model avoids repeatedly predicting the endpoint at intermediate times.

---

## 12. Practical notes

### Time embeddings

Use low-frequency time embeddings. High-frequency Fourier embeddings can destabilize JVP losses because the loss differentiates through time.

### Normalization

Avoid LayerNorm in the JVP-trained student. Since the objective differentiates the model with respect to its inputs, normalization layers change the Jacobian and can destabilize training.

### Output head

Use an unbounded \(f_\theta\) head initially:

\[
C_\theta(z,t,s)=z+(s-t)f_\theta(z,t,s).
\]

Clamp only during sampling if needed.

### Recommended first configuration

Start with teacher-based EMD only:

\[
\mathcal L=\mathcal L_{\text{EMD}}.
\]

Then optionally try:

\[
\mathcal L
=
\mathcal L_{\text{EMD}}
+
\lambda_{\text{LMD}}\mathcal L_{\text{LMD}}
+
\lambda_{\text{anchor}}\mathcal L_{\text{anchor}}.
\]

A reasonable starting point:

```python
train_mode = "emd"
emd_weight = 1.0
lmd_weight = 0.0
pairwise_anchor_weight = 0.0
teacher_interval_anchor_weight = 0.0
jvp_dt_ref = 0.02
```

---

## 13. Core formulation

The final bounded-latent AYF-style RF formulation is:

\[
z_0\sim\mathcal U([-1,1]^d),
\qquad
z_1=E(x)\in[-1,1]^d,
\]

\[
z_t=(1-t)z_0+t z_1,
\]

\[
C_\theta(z,t,s)=z+(s-t)f_\theta(z,t,s),
\]

\[
\mathcal L_{\text{EMD}}
=
\left\|
\operatorname{JVP}_{(z,t)}
C_\theta(z,t,s)
[
(v_\phi(z,t),1)
]
\right\|^2.
\]

This is the bounded-uniform RF analogue of an AYF-style arbitrary-time flow map.
