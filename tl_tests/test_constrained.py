import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from torch.distributions import Normal
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

def savefig(name):
    plt.savefig(os.path.join(PLOT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close()

# ── Toy function ──────────────────────────────────────────────────────────────
def stoich_function(x, y, n, sigma=0.3, A=1.0):
    x0, y0 = 2/n, (n+1)/n
    return A * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def toy_function(x, y, z, sigma=0.3, A=2.5, noise_scale=0.0):
    n = int(z)
    signal = stoich_function(x, y, n, sigma=sigma, A=A)
    if noise_scale > 0:
        raw_noise = np.random.normal(0, noise_scale * A, size=np.shape(signal))
        noise = gaussian_filter(raw_noise, sigma=2)
        return signal + noise
    return signal

# ── Pool ──────────────────────────────────────────────────────────────────────
GRID       = 40
BOUNDS     = torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float64)
g          = np.linspace(0, 3, GRID)
xx, yy     = np.meshgrid(g, g)
xy_pool    = np.stack([xx.ravel(), yy.ravel()], axis=1)

n1_vals    = np.array([toy_function(x, y, 1) for x, y in xy_pool])
n2_vals    = np.array([toy_function(x, y, 2) for x, y in xy_pool])
n3_vals    = np.array([toy_function(x, y, 3) for x, y in xy_pool])

X_pool     = torch.tensor(xy_pool, dtype=torch.float64)
X_norm     = normalize(X_pool, BOUNDS)
n1_raw     = torch.tensor(n1_vals, dtype=torch.float64).unsqueeze(-1)
n2_raw     = torch.tensor(n2_vals, dtype=torch.float64).unsqueeze(-1)
n3_raw     = torch.tensor(n3_vals, dtype=torch.float64).unsqueeze(-1)

# Standardise
n1_mean, n1_std = n1_raw.mean(), n1_raw.std().clamp(min=1e-6)
n2_mean, n2_std = n2_raw.mean(), n2_raw.std().clamp(min=1e-6)
n3_mean, n3_std = n3_raw.mean(), n3_raw.std().clamp(min=1e-6)
n1_std_pool = (n1_raw - n1_mean) / n1_std
n2_std_pool = (n2_raw - n2_mean) / n2_std
n3_std_pool = (n3_raw - n3_mean) / n3_std

# ── Pretrain a frozen N1 GP on the full pool (ground truth proxy) ─────────────
# In a real experiment this would be your N1 BO result.
# Here we give it 20 random points so it has a decent model of N1.
rng         = np.random.default_rng(0)
n1_idx      = rng.choice(len(xy_pool), size=20, replace=False)
X_n1_tr     = X_norm[n1_idx]
y_n1_tr     = n1_std_pool[n1_idx]

gp_n1 = SingleTaskGP(X_n1_tr, y_n1_tr)
fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_n1.likelihood, gp_n1))
gp_n1.eval()

# N1 target = its peak value (standardised)
# n1_peak_raw  = toy_function(2.0, 2.0, 1)           # ≈ 2.5
# target_n1    = (n1_peak_raw - n1_mean.item()) / n1_std.item()
tolerance    = 0.5 / n1_std.item()                  # within 0.5 raw units of target

n1_peak_raw       = toy_function(2.0, 2.0, 1)           # ≈ 2.5
min_threshold_raw = 0.1 * n1_peak_raw  # only exclude truly zero-N1 regions         
min_threshold     = (min_threshold_raw - n1_mean.item()) / n1_std.item()

# ── Custom CEI acquisition ────────────────────────────────────────────────────
def ei_scores(gp, X, best_f):
    """Standard EI in closed form."""
    with torch.no_grad():
        post  = gp.posterior(X)
        mu    = post.mean.squeeze(-1)
        sigma = post.variance.squeeze(-1).sqrt().clamp(min=1e-6)
    z    = (mu - best_f) / sigma
    dist = Normal(torch.zeros_like(z), torch.ones_like(z))
    return sigma * (dist.log_prob(z).exp() + z * dist.cdf(z))

def feasibility_weights(gp_n1, X):
    with torch.no_grad():
        post = gp_n1.posterior(X)
        mu   = post.mean.squeeze(-1)
    # soft weight: higher N1 mean = higher weight
    mu_min = mu.min()
    mu_max = mu.max()
    return (mu - mu_min) / (mu_max - mu_min + 1e-6)

def custom_cei(gp_obj, constraint_gps, X, best_f):
    ei  = ei_scores(gp_obj, X, best_f)
    phi = torch.ones(X.shape[0], dtype=X.dtype)
    for gp_c in constraint_gps:
        phi = phi * feasibility_weights(gp_c, X)
    return ei * phi

# ── BO loop: Plain EI vs Custom CEI ──────────────────────────────────────────
N_INIT = 2
N_ITER = 40

# Adversarial seeds: farthest from N2 peak (1, 1.5)
n2_peak   = np.array([1.0, 1.5])
dists     = np.linalg.norm(xy_pool - n2_peak, axis=1)
init_idx  = np.argsort(dists)[-N_INIT:]

results = {}

for method in ['plain_ei', 'custom_cei']:
    queried    = set(init_idx.tolist())
    X_tr       = X_norm[list(queried)]
    y_tr       = n2_std_pool[list(queried)]
    best_vals  = []
    selected   = []

    print(f"\n[{method}] true N2 max = {n2_raw.max().item():.4f}")

    for i in range(N_ITER):
        gp_n2 = SingleTaskGP(X_tr, y_tr)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_n2.likelihood, gp_n2))
        gp_n2.eval()

        if method == 'plain_ei':
            scores = ei_scores(gp_n2, X_norm, best_f=y_tr.max())
        else:
            scores = custom_cei(gp_n2, [gp_n1], X_norm, best_f=y_tr.max())

        for idx in queried:
            scores[idx] = -float('inf')

        best_idx = scores.argmax().item()
        queried.add(best_idx)
        selected.append(best_idx)
        X_tr = torch.cat([X_tr, X_norm[best_idx].unsqueeze(0)])
        y_tr = torch.cat([y_tr, n2_std_pool[best_idx].unsqueeze(0)])

        best_raw = n2_raw[list(queried)].max().item()
        best_vals.append(best_raw)
        print(f"  iter {i+1:3d} | idx {best_idx:4d} | "
              f"n2: {n2_raw[best_idx].item():.4f} | best: {best_raw:.4f}")

    results[method] = {'best': best_vals, 'selected': selected}

# ── Convergence plot ──────────────────────────────────────────────────────────
iters = np.arange(1, N_ITER + 1)
plt.figure(figsize=(7, 4))
plt.plot(iters, results['plain_ei']['best'],   marker='o', label='Plain EI')
plt.plot(iters, results['custom_cei']['best'], marker='s', label='Custom CEI')
plt.axhline(n2_raw.max().item(), color='r', linestyle='--',
            label=f"true max ({n2_raw.max().item():.4f})")
plt.xlabel('Iteration')
plt.ylabel('Best N2 (raw)')
plt.title('Plain EI vs Custom CEI with N1 feasibility weighting')
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig('results/n2_convergence.png')

# ── Scatter overlays ──────────────────────────────────────────────────────────
n2_landscape = n2_raw.numpy().reshape(GRID, GRID)
seed_xy      = xy_pool[list(init_idx)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, method in zip(axes, ['plain_ei', 'custom_cei']):
    sel   = results[method]['selected']
    bo_xy = xy_pool[sel]
    ax.imshow(n2_landscape, origin='lower', extent=(0,3,0,3), cmap='viridis')
    sc = ax.scatter(bo_xy[:,0], bo_xy[:,1],
                    c=np.arange(1, len(bo_xy)+1), cmap='plasma',
                    edgecolors='white', linewidths=0.5, s=60, zorder=3)
    for k, (x, y) in enumerate(bo_xy):
        ax.text(x, y, str(k+1), fontsize=6, ha='center', va='center',
                color='white', zorder=4)
    ax.scatter(seed_xy[:,0], seed_xy[:,1],
               marker='*', s=150, color='cyan', edgecolors='black',
               linewidths=0.5, zorder=4)
    ax.set_title(method.replace('_', ' ').title())
    ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle('N2 queried points')
plt.tight_layout()
savefig('results/n2_scatter.png')

# ── Feasibility weight map ────────────────────────────────────────────────────
# Shows where the N1 GP thinks the target is reachable
phi_map = feasibility_weights(gp_n1, X_norm)
print(f"φ > 0.1 at {(phi_map > 0.1).sum().item()} / {len(phi_map)} pool points")
print(f"φ > 0.5 at {(phi_map > 0.5).sum().item()} / {len(phi_map)} pool points")
phi_grid = phi_map.numpy().reshape(GRID, GRID)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(n2_landscape, origin='lower', extent=(0,3,0,3), cmap='viridis')
axes[0].set_title('N2 landscape (objective)')
axes[1].imshow(phi_grid, origin='lower', extent=(0,3,0,3), cmap='hot', vmin=0, vmax=1)
axes[1].set_title('φ(x): N1 feasibility weight\n(bright = near N1 target)')
for ax in axes:
    ax.set_xlabel('x'); ax.set_ylabel('y')
plt.tight_layout()
savefig('results/n1_feasibility_map.png')

# ── Plot GP uncertainty maps (N1 and final N2 for each method) ────────────────
with torch.no_grad():
    post_n1 = gp_n1.posterior(X_norm)
    n1_std_map = post_n1.variance.squeeze(-1).sqrt().numpy().reshape(GRID, GRID)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(n1_std_map, origin='lower', extent=(0,3,0,3), cmap='magma')
axes[0].set_title('N1 GP std')

for ax, method in zip(axes[1:], ['plain_ei', 'custom_cei']):
    sel_idx = results[method]['selected']
    X_sel = X_norm[sel_idx]
    y_sel = n2_std_pool[sel_idx]
    gp_final = SingleTaskGP(X_sel, y_sel)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_final.likelihood, gp_final))
    gp_final.eval()
    with torch.no_grad():
        post = gp_final.posterior(X_norm)
    std_map = post.variance.squeeze(-1).sqrt().numpy().reshape(GRID, GRID)
    ax.imshow(std_map, origin='lower', extent=(0,3,0,3), cmap='magma')
    ax.set_title(f'N2 GP std ({method})')

for ax in axes:
    ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle('GP predictive standard deviation maps')
plt.tight_layout()
savefig('results/gp_std_maps.png')


# ══════════════════════════════════════════════════════════════════════════════
# N3 BO: optimise N3 with N1 and N2 GPs as constraints
# ══════════════════════════════════════════════════════════════════════════════

# Build a frozen N2 GP from the N2 CEI results (cascade: N1→N2→N3).
# This represents what we learned about N2 from the constrained N2 experiment.
n2_cei_sel  = results['custom_cei']['selected']
X_n2_frozen = torch.cat([X_norm[list(init_idx)], X_norm[n2_cei_sel]])
y_n2_frozen = torch.cat([n2_std_pool[list(init_idx)], n2_std_pool[n2_cei_sel]])
gp_n2_frozen = SingleTaskGP(X_n2_frozen, y_n2_frozen)
fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_n2_frozen.likelihood, gp_n2_frozen))
gp_n2_frozen.eval()

# N3 peak: layer 3 → x0=2/3≈0.67, y0=4/3≈1.33
n3_peak  = np.array([2/3, 4/3])
dists_n3 = np.linalg.norm(xy_pool - n3_peak, axis=1)
init_idx_n3 = np.argsort(dists_n3)[-N_INIT:]

n3_results = {}

for method in ['plain_ei', 'n3_cei']:
    queried   = set(init_idx_n3.tolist())
    X_tr      = X_norm[list(queried)]
    y_tr      = n3_std_pool[list(queried)]
    best_vals = []
    selected  = []

    print(f"\n[{method}] true N3 max = {n3_raw.max().item():.4f}")

    for i in range(N_ITER):
        gp_n3 = SingleTaskGP(X_tr, y_tr)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_n3.likelihood, gp_n3))
        gp_n3.eval()

        if method == 'plain_ei':
            scores = ei_scores(gp_n3, X_norm, best_f=y_tr.max())
        else:
            scores = custom_cei(gp_n3, [gp_n1, gp_n2_frozen], X_norm, best_f=y_tr.max())

        for idx in queried:
            scores[idx] = -float('inf')

        best_idx = scores.argmax().item()
        queried.add(best_idx)
        selected.append(best_idx)
        X_tr = torch.cat([X_tr, X_norm[best_idx].unsqueeze(0)])
        y_tr = torch.cat([y_tr, n3_std_pool[best_idx].unsqueeze(0)])

        best_raw = n3_raw[list(queried)].max().item()
        best_vals.append(best_raw)
        print(f"  iter {i+1:3d} | idx {best_idx:4d} | "
              f"n3: {n3_raw[best_idx].item():.4f} | best: {best_raw:.4f}")

    n3_results[method] = {'best': best_vals, 'selected': selected}

# ── N3 convergence plot ───────────────────────────────────────────────────────
plt.figure(figsize=(7, 4))
plt.plot(iters, n3_results['plain_ei']['best'], marker='o', label='Plain EI')
plt.plot(iters, n3_results['n3_cei']['best'],   marker='s', label='CEI (N1+N2 constraints)')
plt.axhline(n3_raw.max().item(), color='r', linestyle='--',
            label=f"true max ({n3_raw.max().item():.4f})")
plt.xlabel('Iteration')
plt.ylabel('Best N3 (raw)')
plt.title('N3 BO: Plain EI vs CEI with N1+N2 constraints')
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig('results/n3_convergence.png')

# ── N3 scatter overlays ───────────────────────────────────────────────────────
n3_landscape = n3_raw.numpy().reshape(GRID, GRID)
seed_xy_n3   = xy_pool[list(init_idx_n3)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, method in zip(axes, ['plain_ei', 'n3_cei']):
    sel   = n3_results[method]['selected']
    bo_xy = xy_pool[sel]
    ax.imshow(n3_landscape, origin='lower', extent=(0,3,0,3), cmap='viridis')
    ax.scatter(bo_xy[:,0], bo_xy[:,1],
               c=np.arange(1, len(bo_xy)+1), cmap='plasma',
               edgecolors='white', linewidths=0.5, s=60, zorder=3)
    for k, (x, y) in enumerate(bo_xy):
        ax.text(x, y, str(k+1), fontsize=6, ha='center', va='center',
                color='white', zorder=4)
    ax.scatter(seed_xy_n3[:,0], seed_xy_n3[:,1],
               marker='*', s=150, color='cyan', edgecolors='black',
               linewidths=0.5, zorder=4)
    label = 'Plain EI' if method == 'plain_ei' else 'CEI (N1+N2 constraints)'
    ax.set_title(label)
    ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle('N3 queried points')
plt.tight_layout()
savefig('results/n3_scatter.png')

# ── Joint feasibility map: N1 × N2 constraint weight ─────────────────────────
phi_n1 = feasibility_weights(gp_n1,       X_norm)
phi_n2 = feasibility_weights(gp_n2_frozen, X_norm)
phi_joint = (phi_n1 * phi_n2).numpy().reshape(GRID, GRID)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(phi_n1.numpy().reshape(GRID, GRID),
               origin='lower', extent=(0,3,0,3), cmap='hot', vmin=0, vmax=1)
axes[0].set_title('φ N1 weight')
axes[1].imshow(phi_n2.numpy().reshape(GRID, GRID),
               origin='lower', extent=(0,3,0,3), cmap='hot', vmin=0, vmax=1)
axes[1].set_title('φ N2 weight')
axes[2].imshow(phi_joint,
               origin='lower', extent=(0,3,0,3), cmap='hot', vmin=0, vmax=1)
axes[2].set_title('φ joint (N1 × N2)')
for ax in axes:
    ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle('Feasibility weights for N3 CEI')
plt.tight_layout()
savefig('results/n3_feasibility_joint.png')
