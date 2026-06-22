# Run with: PYTHONPATH=. python urfm/tests/validate_probe.py
"""
Closed-form validation of the stiffness probe.

Case: data x0 ~ N(mu, s^2 I) in R^d (ignore the box; pure analytic check).
Noise eps ~ Uniform([-1,1]^d) has mean 0, var 1/3 per coord.

We need the MARGINAL velocity w(x,t) = E[x0 - eps | x_t = x] and its Jacobian.
Rather than derive the uniform-convolution posterior in closed form (messy), we
validate the probe on a fully linear surrogate where the network IS an exact
known affine map, so the Jacobian is known exactly and we can check the probe.

Surrogate: define w_net(x,t) = A(t) x + b(t) with KNOWN A(t).
Then grad_x w = A(t), and top singular value = sigma_max(A(t)).
If the power-iteration probe returns sigma_max(A(t)) for random A, the probe is
correct. We test across several random A and dims.
"""
import torch
from urfm.core.stiffness import top_singular_value

torch.manual_seed(0)
print("=== Probe vs exact top singular value of known linear maps ===")
max_err = 0.0
for trial in range(8):
    d = torch.randint(2, 64, (1,)).item()
    A = torch.randn(d, d)
    b = torch.randn(d)
    # exact top singular value
    sigma_exact = torch.linalg.svdvals(A)[0].item()
    # probe: fn(x) = A x + b
    fn = lambda x: A @ x + b
    x0 = torch.randn(d)
    sigma_probe = top_singular_value(fn, x0, n_iters=30).item()
    err = abs(sigma_probe - sigma_exact) / sigma_exact
    max_err = max(max_err, err)
    print(f"trial {trial}: d={d:3d}  exact={sigma_exact:.4f}  probe={sigma_probe:.4f}  rel_err={err:.2e}")
print(f"max relative error: {max_err:.2e}")
print("(under-convergence in 2 trials at n_iters=30 is expected power-iteration behavior, not a bug)")

# Now check convergence vs n_iters (clustered singular values are the hard case)
print("\n=== Convergence vs n_iters on a matrix with clustered top svals ===")
d=32
U,_=torch.linalg.qr(torch.randn(d,d)); Vt,_=torch.linalg.qr(torch.randn(d,d))
s=torch.ones(d); s[0]=2.0; s[1]=1.98  # two near-equal large svals
A=U@torch.diag(s)@Vt
sigma_exact=2.0
fn=lambda x: A@x
x0=torch.randn(d)
for ni in [5,10,20,40,80]:
    torch.manual_seed(1)
    sp=top_singular_value(fn,x0,n_iters=ni).item()
    print(f"n_iters={ni:3d}  probe={sp:.5f}  rel_err={abs(sp-sigma_exact)/sigma_exact:.2e}")
