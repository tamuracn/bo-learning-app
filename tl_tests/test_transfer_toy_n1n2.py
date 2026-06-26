"""
Benchmark: Plain EI  vs  Hard-Mask EI  vs  ConstrainedExpectedImprovement (CEI)
=================================================================================
All three N2 methods optionally use a frozen N1 GP as a constraint.

Method A — Plain EI:
    Standard Expected Improvement on the N2 GP with no constraint knowledge.
    Baseline that controls for acquisition function choice.

Method B — Hard-Mask EI:
    Score all pool points with EI on the N2 GP, then zero-out any point
    where the N1 GP's upper confidence bound exceeds the threshold.
    Fair comparison to CEI: same acquisition function, different constraint
    mechanism.

Method C — CEI (ConstrainedExpectedImprovement):
    Stack N1 and N2 GPs into a ModelListGP. CEI jointly maximises
    expected improvement on N2 while weighting by P(N1 <= threshold).

All three methods share the same N2 seed points so the only variable is
the acquisition / constraint strategy.

Notes on design choices
-----------------------
* The N1 GP is fit once on the N1 BO data and then **frozen** for all N2
  iterations. This is intentional: it models N1 as fixed prior information
  (static transfer), not a joint or online model.
* N1 BO seeds are chosen randomly (not anti-podally) so the N1 GP has
  coverage in the *feasible* low-N1 region as well as near the peak.
  Seeding farthest from the peak was the previous behaviour; it left the
  GP with poor uncertainty quantification exactly where the constraint
  boundary matters most for N2 acquisition.
* UCB_BETA appears only once (config block). The duplicate definition that
  silently overwrote it has been removed.

Outputs
-------
    - Console: per-iteration best values for all three methods
    - Plots:   convergence curves + scatter overlays + GP posterior grids
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
from torch.distributions import Normal
warnings.filterwarnings("ignore")

from toy_model.toy_function import toy_function

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ConstrainedExpectedImprovement, ExpectedImprovement
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TOY_N1_LAYER   = 1          # stoich layer for N1 (peak at x=2, y=2)
TOY_N2_LAYER   = 2          # stoich layer for N2 (peak at x=1, y=1.5)
TOY_POOL_GRID  = 50         # pool grid: TOY_POOL_GRID × TOY_POOL_GRID points
TOY_BOUNDS     = torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float64)
TOY_N1_THRESHOLD_REAL = 2.0 # N1 values ABOVE this are "infeasible" for N2 queries
                             # (CEI constraint: N1 <= threshold)
TOY_SIGMA      = 0.5        # peak width for toy_function

# Hard-mask confidence level for Method B:
# UCB percentile used to build the N1 fence before EI scoring.
# 1.28 ≈ 90th pct, 1.645 ≈ 95th pct, 1.96 ≈ 97.5th pct
CONSTRAINT_CONFIDENCE = 1.28

# Single definition of UCB_BETA (removed duplicate that previously appeared
# again in the N1 BO block, silently overwriting this value).
UCB_BETA = 2.0              # exploration weight — kept for reference / future use

N1_INIT_POINTS = 2
MAX_ITER_N1    = 50
N2_INIT_POINTS = 2
MAX_ITER_N2    = 50
CONVERGE_TOL   = 0.01       # stop when best >= (1 - tol) * true pool max


# ══════════════════════════════════════════════════════════════════════════════
# Generate pools
# ══════════════════════════════════════════════════════════════════════════════
print("Generating toy data...")
g       = np.linspace(0, 3, TOY_POOL_GRID)
xx, yy  = np.meshgrid(g, g)
xy_pool = np.stack([xx.ravel(), yy.ravel()], axis=1)

n1_z_pool = np.array([toy_function(x, y, TOY_N1_LAYER, sigma=TOY_SIGMA, noise_scale=0.0)
                       for x, y in xy_pool])
n2_z_pool = np.array([toy_function(x, y, TOY_N2_LAYER, sigma=TOY_SIGMA, noise_scale=0.0)
                       for x, y in xy_pool])

X_pool_raw  = torch.tensor(xy_pool,    dtype=torch.float64)
n1_pool_raw = torch.tensor(n1_z_pool,  dtype=torch.float64).unsqueeze(-1)
n2_pool_raw = torch.tensor(n2_z_pool,  dtype=torch.float64).unsqueeze(-1)

X_pool_norm = normalize(X_pool_raw, TOY_BOUNDS)

# Standardise using pool statistics
y1_mean     = n1_pool_raw.mean()
y1_std      = n1_pool_raw.std().clamp(min=1e-6)
n1_pool_std = (n1_pool_raw - y1_mean) / y1_std

y2_mean     = n2_pool_raw.mean()
y2_std      = n2_pool_raw.std().clamp(min=1e-6)
n2_pool_std = (n2_pool_raw - y2_mean) / y2_std

true_max_n1 = n1_pool_raw.max().item()
true_max_n2 = n2_pool_raw.max().item()

n1_landscape = n1_pool_raw.numpy().reshape(TOY_POOL_GRID, TOY_POOL_GRID)
n2_landscape = n2_pool_raw.numpy().reshape(TOY_POOL_GRID, TOY_POOL_GRID)


# ══════════════════════════════════════════════════════════════════════════════
# N1 BO: build the frozen constraint GP
# ══════════════════════════════════════════════════════════════════════════════
# FIX: seed N1 BO randomly rather than antipodally from the N1 peak.
# The previous strategy (farthest-from-peak seeds) left the N1 GP with poor
# coverage in the low-N1 / feasible region — exactly where the constraint
# boundary matters for N2 acquisition.
rng_n1       = np.random.default_rng(40)
init_indices = rng_n1.choice(len(xy_pool), size=N1_INIT_POINTS, replace=False)

queried      = set(init_indices.tolist())
X_train_n1   = X_pool_norm[list(queried)]
y_train_n1   = n1_pool_std[list(queried)]

print(f"\nPool-based EI on N1 | init={N1_INIT_POINTS} random pts | true max={true_max_n1:.4f}")
best_n1_values   = []
selected_n1      = []

for i in range(MAX_ITER_N1):
    gp_n1 = SingleTaskGP(X_train_n1, y_train_n1)
    mll   = ExactMarginalLogLikelihood(gp_n1.likelihood, gp_n1)
    fit_gpytorch_mll(mll)
    gp_n1.eval()

    with torch.no_grad():
        acq_vals = ExpectedImprovement(gp_n1, best_f=y_train_n1.max())(
            X_pool_norm.unsqueeze(1))
    for idx in queried:
        acq_vals[idx] = -float('inf')

    best_idx = acq_vals.argmax().item()
    queried.add(best_idx)
    selected_n1.append(best_idx)

    X_train_n1 = torch.cat([X_train_n1, X_pool_norm[best_idx].unsqueeze(0)], dim=0)
    y_train_n1 = torch.cat([y_train_n1, n1_pool_std[best_idx].unsqueeze(0)],  dim=0)

    best_raw = n1_pool_raw[list(queried)].max().item()
    best_n1_values.append(best_raw)
    print(f"  Iter {i+1:3d} | pool idx: {best_idx:4d} | "
          f"y_n1: {n1_pool_raw[best_idx].item():.4f} | best: {best_raw:.4f} / {true_max_n1:.4f}")

# ── N1 convergence plot ───────────────────────────────────────────────────────
plt.figure()
plt.plot(np.arange(1, len(best_n1_values) + 1), best_n1_values, marker='o')
plt.axhline(true_max_n1, color='r', linestyle='--', label=f'true max ({true_max_n1:.4f})')
plt.xlabel('Iteration')
plt.ylabel('Best N1 value (raw)')
plt.title('Pool-based EI on N1: convergence from random seed points')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ── N1 scatter overlay ────────────────────────────────────────────────────────
seed_xy_n1 = xy_pool[list(init_indices)]
bo_xy_n1   = xy_pool[selected_n1]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(n1_landscape, origin='lower', extent=(0, 3, 0, 3), cmap='viridis')
fig.colorbar(im, ax=ax, label='N1 value (raw)')
sc = ax.scatter(bo_xy_n1[:, 0], bo_xy_n1[:, 1],
                c=np.arange(1, len(bo_xy_n1) + 1), cmap='plasma',
                edgecolors='white', linewidths=0.5, s=60, zorder=3, label='BO selected')
fig.colorbar(sc, ax=ax, label='Iteration')
for k, (x, y) in enumerate(bo_xy_n1):
    ax.text(x, y, str(k + 1), fontsize=6, ha='center', va='center', color='white', zorder=4)
ax.scatter(seed_xy_n1[:, 0], seed_xy_n1[:, 1],
           marker='*', s=150, color='cyan', edgecolors='black',
           linewidths=0.5, zorder=4, label='Seed (random)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('N1 pool: BO-selected points (numbered by iteration)')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()

# ── Fit & freeze the final N1 GP ─────────────────────────────────────────────
# This GP is intentionally static: it represents fixed prior knowledge from N1
# and is not updated during N2 BO iterations (static transfer).
gp_n1_final = SingleTaskGP(X_train_n1, y_train_n1)
fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_n1_final.likelihood, gp_n1_final))
gp_n1_final.eval()

# Threshold in standardised N1 space (CEI operates on standardised values)
# Constraint direction: N1 <= threshold  →  feasible region is LOW N1.
# High N1 is "infeasible" for N2 queries.
threshold_n1_std = (TOY_N1_THRESHOLD_REAL - y1_mean) / y1_std

# ── N1 GP posterior plots ─────────────────────────────────────────────────────
with torch.no_grad():
    post_n1   = gp_n1_final.posterior(X_pool_norm)
    mean_n1   = (post_n1.mean.squeeze(-1) * y1_std + y1_mean).numpy().reshape(TOY_POOL_GRID, TOY_POOL_GRID)
    std_n1    = (post_n1.variance.squeeze(-1).sqrt() * y1_std).numpy().reshape(TOY_POOL_GRID, TOY_POOL_GRID)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(mean_n1, origin='lower', extent=(0, 3, 0, 3))
axes[0].set_title('N1 GP posterior mean (raw)')
fig.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(std_n1, origin='lower', extent=(0, 3, 0, 3))
axes[1].set_title('N1 GP posterior std (raw)')
fig.colorbar(im1, ax=axes[1])
for ax in axes:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# N2 Benchmark: Plain EI  vs  Hard-Mask EI  vs  CEI
# ══════════════════════════════════════════════════════════════════════════════
# Shared random seeds — all three methods start from the same N2_INIT_POINTS
# observations so the only variable is the acquisition / constraint strategy.
# rng_n2      = np.random.default_rng(7)
# init_idx_n2 = rng_n2.choice(len(xy_pool), size=N2_INIT_POINTS, replace=False)
# seed_xy_n2  = xy_pool[init_idx_n2]
n2_peak     = np.array([1.0, 1.5])
dists_n2    = np.linalg.norm(xy_pool - n2_peak, axis=1)
init_idx_n2 = np.argsort(dists_n2)[-N2_INIT_POINTS:]
seed_xy_n2  = xy_pool[init_idx_n2]

iters = np.arange(1, MAX_ITER_N2 + 1)


# ── Helper: GP posterior grids in raw N2 scale ────────────────────────────────
def gp_posterior_grids(X_tr, y_tr):
    gp = SingleTaskGP(X_tr, y_tr)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()
    with torch.no_grad():
        post = gp.posterior(X_pool_norm)
    mean = (post.mean.squeeze(-1) * y2_std + y2_mean).numpy().reshape(TOY_POOL_GRID, TOY_POOL_GRID)
    std  = (post.variance.squeeze(-1).sqrt() * y2_std).numpy().reshape(TOY_POOL_GRID, TOY_POOL_GRID)
    return mean, std


# ── Method A: Plain EI (no constraint) ───────────────────────────────────────
print(f"\n[Method A — Plain EI] N2 BO | true max={true_max_n2:.4f}")
queried_plain  = set(init_idx_n2.tolist())
X_tr_plain     = X_pool_norm[list(queried_plain)]
y_tr_plain     = n2_pool_std[list(queried_plain)]
best_plain     = []
selected_plain = []

for i in range(MAX_ITER_N2):
    gp = SingleTaskGP(X_tr_plain, y_tr_plain)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()

    with torch.no_grad():
        acq_vals = ExpectedImprovement(gp, best_f=y_tr_plain.max())(
            X_pool_norm.unsqueeze(1))
    for idx in queried_plain:
        acq_vals[idx] = -float('inf')

    best_idx = acq_vals.argmax().item()
    queried_plain.add(best_idx)
    selected_plain.append(best_idx)

    X_tr_plain = torch.cat([X_tr_plain, X_pool_norm[best_idx].unsqueeze(0)], dim=0)
    y_tr_plain = torch.cat([y_tr_plain, n2_pool_std[best_idx].unsqueeze(0)],  dim=0)

    best_raw = n2_pool_raw[list(queried_plain)].max().item()
    best_plain.append(best_raw)
    print(f"  Iter {i+1:3d} | idx: {best_idx:4d} | "
          f"y_n2: {n2_pool_raw[best_idx].item():.4f} | best: {best_raw:.4f}")


# ── Method B: Hard-Mask EI ────────────────────────────────────────────────────
# FIX: this method was described in the docstring but not implemented.
# It provides the critical "fair" baseline: same EI acquisition as Method A,
# but with N1 constraint knowledge applied via a hard mask. This isolates the
# effect of the constraint from the choice of acquisition function when
# comparing against CEI (Method C).
#
# A candidate is masked out if the N1 GP's UCB exceeds the threshold,
# i.e. we only query where we are confident N1 is feasible.
print(f"\n[Method B — Hard-Mask EI] N2 BO | true max={true_max_n2:.4f}")
queried_mask  = set(init_idx_n2.tolist())
X_tr_mask     = X_pool_norm[list(queried_mask)]
y_tr_mask     = n2_pool_std[list(queried_mask)]
best_mask     = []
selected_mask = []

for i in range(MAX_ITER_N2):
    # Fit N2 GP
    gp_n2 = SingleTaskGP(X_tr_mask, y_tr_mask)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_n2.likelihood, gp_n2))
    gp_n2.eval()

    with torch.no_grad():
        # EI scores for all pool points
        acq_vals = ExpectedImprovement(gp_n2, best_f=y_tr_mask.max())(
            X_pool_norm.unsqueeze(1))

        # Build hard mask from N1 GP: zero out points where
        # N1 UCB > threshold (upper confidence bound exceeds feasibility limit)
        post_n1    = gp_n1_final.posterior(X_pool_norm)
        n1_mean    = post_n1.mean.squeeze(-1)
        n1_std_vec = post_n1.variance.squeeze(-1).sqrt()
        n1_ucb     = n1_mean + CONSTRAINT_CONFIDENCE * n1_std_vec
        infeasible = n1_ucb > threshold_n1_std

    acq_vals[infeasible] = -float('inf')

    for idx in queried_mask:
        acq_vals[idx] = -float('inf')

    best_idx = acq_vals.argmax().item()

    # Fallback: if the mask eliminates all candidates, pick the unconstrained best
    if acq_vals.max().item() == -float('inf'):
        print(f"  Iter {i+1:3d} | WARNING: all candidates masked — falling back to unconstrained EI")
        with torch.no_grad():
            acq_vals_fb = ExpectedImprovement(gp_n2, best_f=y_tr_mask.max())(
                X_pool_norm.unsqueeze(1))
        for idx in queried_mask:
            acq_vals_fb[idx] = -float('inf')
        best_idx = acq_vals_fb.argmax().item()

    queried_mask.add(best_idx)
    selected_mask.append(best_idx)

    X_tr_mask = torch.cat([X_tr_mask, X_pool_norm[best_idx].unsqueeze(0)], dim=0)
    y_tr_mask = torch.cat([y_tr_mask, n2_pool_std[best_idx].unsqueeze(0)],  dim=0)

    best_raw    = n2_pool_raw[list(queried_mask)].max().item()
    n_feasible  = (~infeasible).sum().item()
    best_mask.append(best_raw)
    print(f"  Iter {i+1:3d} | idx: {best_idx:4d} | "
          f"y_n2: {n2_pool_raw[best_idx].item():.4f} | "
          f"feasible pts: {n_feasible}/{len(xy_pool)} | best: {best_raw:.4f}")


# ── Method C: CEI with N1 transfer ───────────────────────────────────────────
print(f"\n[Method C — CEI Transfer] N2 BO | true max={true_max_n2:.4f}")
queried_cei  = set(init_idx_n2.tolist())
X_tr_cei     = X_pool_norm[list(queried_cei)]
y_tr_cei     = n2_pool_std[list(queried_cei)]
best_cei     = []
selected_cei = []

for i in range(MAX_ITER_N2):
    gp_n2 = SingleTaskGP(X_tr_cei, y_tr_cei)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_n2.likelihood, gp_n2))
    gp_n2.eval()

    # CEI: maximise EI on N2 (index 0) subject to N1 <= threshold (index 1).
    # Constraint direction: (None, threshold_n1_std) means N1 <= threshold,
    # i.e. the feasible region is where N1 is LOW. High N1 is infeasible.
    cei = ConstrainedExpectedImprovement(
        model=ModelListGP(gp_n2, gp_n1_final),
        best_f=y_tr_cei.max(),
        objective_index=0,
        constraints={1: (None, threshold_n1_std)},
    )

    with torch.no_grad():
        acq_vals = cei(X_pool_norm.unsqueeze(1))
    for idx in queried_cei:
        acq_vals[idx] = -float('inf')

    best_idx = acq_vals.argmax().item()
    queried_cei.add(best_idx)
    selected_cei.append(best_idx)

    X_tr_cei = torch.cat([X_tr_cei, X_pool_norm[best_idx].unsqueeze(0)], dim=0)
    y_tr_cei = torch.cat([y_tr_cei, n2_pool_std[best_idx].unsqueeze(0)],  dim=0)

    best_raw = n2_pool_raw[list(queried_cei)].max().item()
    best_cei.append(best_raw)

    with torch.no_grad():
        post_sel = gp_n1_final.posterior(X_pool_norm[best_idx].unsqueeze(0))
        pof = Normal(post_sel.mean.item(),
                     post_sel.variance.sqrt().item()).cdf(
                         torch.tensor(threshold_n1_std)).item()
    print(f"  Iter {i+1:3d} | idx: {best_idx:4d} | "
          f"y_n2: {n2_pool_raw[best_idx].item():.4f} | "
          f"P(N1<=thr): {pof:.3f} | best: {best_raw:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

# ── Convergence comparison ────────────────────────────────────────────────────
plt.figure(figsize=(7, 4))
plt.plot(iters, best_plain, marker='o', label='Method A: Plain EI')
plt.plot(iters, best_mask,  marker='^', label='Method B: Hard-Mask EI')
plt.plot(iters, best_cei,   marker='s', label='Method C: CEI + N1 transfer')
plt.axhline(true_max_n2, color='r', linestyle='--', label=f'true max ({true_max_n2:.4f})')
plt.xlabel('Iteration')
plt.ylabel('Best N2 value (raw)')
plt.title('N2 BO benchmark: Plain EI vs Hard-Mask EI vs CEI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ── Scatter overlays ──────────────────────────────────────────────────────────
def scatter_overlay(ax, title, sel_indices):
    bo_xy = xy_pool[sel_indices]
    im = ax.imshow(n2_landscape, origin='lower', extent=(0, 3, 0, 3), cmap='viridis')
    sc = ax.scatter(bo_xy[:, 0], bo_xy[:, 1],
                    c=np.arange(1, len(bo_xy) + 1), cmap='plasma',
                    edgecolors='white', linewidths=0.5, s=60, zorder=3)
    for k, (x, y) in enumerate(bo_xy):
        ax.text(x, y, str(k + 1), fontsize=6, ha='center', va='center',
                color='white', zorder=4)
    ax.scatter(seed_xy_n2[:, 0], seed_xy_n2[:, 1],
               marker='*', s=150, color='cyan', edgecolors='black',
               linewidths=0.5, zorder=4)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return im, sc

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
scatter_overlay(axes[0], 'Method A: Plain EI',           selected_plain)
scatter_overlay(axes[1], 'Method B: Hard-Mask EI',       selected_mask)
scatter_overlay(axes[2], 'Method C: CEI + N1 transfer',  selected_cei)
plt.suptitle('N2 queried points (numbered by iteration, cyan ★ = seeds)')
plt.tight_layout()
plt.show()


# ── GP posterior grids (3 methods × 3 rows) ──────────────────────────────────
mean_plain, std_plain = gp_posterior_grids(X_tr_plain, y_tr_plain)
mean_mask,  std_mask  = gp_posterior_grids(X_tr_mask,  y_tr_mask)
mean_cei,   std_cei   = gp_posterior_grids(X_tr_cei,   y_tr_cei)

methods = [
    (mean_plain, std_plain, 'Plain EI',          selected_plain),
    (mean_mask,  std_mask,  'Hard-Mask EI',       selected_mask),
    (mean_cei,   std_cei,   'CEI + N1 transfer',  selected_cei),
]

fig, axes = plt.subplots(3, 3, figsize=(12, 9))

for col, (mean, std, label, sel) in enumerate(methods):
    bo_xy = xy_pool[sel]

    def overlay(ax):
        ax.scatter(bo_xy[:, 0], bo_xy[:, 1],
                   c=np.arange(1, len(bo_xy) + 1), cmap='plasma',
                   edgecolors='white', linewidths=0.5, s=30, zorder=3)
        ax.scatter(seed_xy_n2[:, 0], seed_xy_n2[:, 1],
                   marker='*', s=100, color='cyan', edgecolors='black',
                   linewidths=0.5, zorder=4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    ax = axes[0][col]
    im = ax.imshow(n2_landscape, origin='lower', extent=(0, 3, 0, 3), cmap='viridis')
    fig.colorbar(im, ax=ax)
    overlay(ax)
    ax.set_title(f'{label} — true N2')

    for row, (grid, subtitle) in enumerate([(mean, 'posterior mean'), (std, 'posterior std')], start=1):
        ax = axes[row][col]
        im = ax.imshow(grid, origin='lower', extent=(0, 3, 0, 3))
        fig.colorbar(im, ax=ax)
        overlay(ax)
        ax.set_title(f'{label} — {subtitle}')

plt.suptitle('N2 GP results (cyan ★ = seeds, dots numbered by iteration)', y=1.01)
plt.tight_layout()
plt.show()