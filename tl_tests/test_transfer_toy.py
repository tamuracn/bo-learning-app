"""
Benchmark: Hard-Mask (UCB) vs Hard-Mask (EI) vs ConstrainedExpectedImprovement (CEI)
======================================================================================
All three methods use an N1 GP as a constraint to guide N2 data collection.

Method A — Hard Mask + UCB:
    Score all pool points with UCB on the N2 GP, then zero-out any point
    where the N1 GP's upper confidence bound exceeds a threshold.

Method B — ConstrainedExpectedImprovement (CEI):
    Stack N1 and N2 GPs into a ModelListGP. CEI jointly maximises the
    expected improvement on N2 while weighting by P(N1 constraint satisfied).

Method C — Hard Mask + EI:
    Same hard mask as Method A but uses Expected Improvement instead of UCB
    to score the feasible candidates.

Outputs:
    - Console: per-iteration best values for all methods
    - benchmark_results.png: convergence curves + feasibility diagnostics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from torch.distributions import Normal
warnings.filterwarnings("ignore")

from toy_model.toy_function import toy_function

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound, ConstrainedExpectedImprovement, ExpectedImprovement
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these to match your experiment
# ══════════════════════════════════════════════════════════════════════════════

TOY_N1_LAYER   = 1          # stoich layer for N1 (peak at x=2, y=2)
TOY_N2_LAYER   = 2          # stoich layer for N2 (peak at x=1, y=1.5)
TOY_N1_SAMPLES = 60         # number of random N1 training points
TOY_POOL_GRID  = 35         # N2 pool grid: TOY_POOL_GRID × TOY_POOL_GRID points
TOY_BOUNDS     = torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float64)
TOY_N1_THRESHOLD_REAL = 2.0 # N1 values above this are "infeasible" for N2 queries
TOY_SIGMA      = 0.5        # peak width for toy_function (default 0.1 is too sharp
                             # for EI/CEI to find via GP — 0.5 gives a gradient
                             # detectable from anywhere in the space)

# Hard-mask confidence level: UCB percentile used to evaluate the N1 fence
# 1.28 ≈ 90th pct, 1.645 ≈ 95th pct, 1.96 ≈ 97.5th pct
CONSTRAINT_CONFIDENCE = 1.28

N_ITERATIONS = 50           # BO iterations per method
N_SEEDS      = 5            # repeated runs to average (set to 1 for quick test)
UCB_BETA     = 2.0          # exploration weight for N2 UCB (Method A)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def fit_n1_gp(X_n1, y_n1):
    """Fit and freeze the N1 GP (donor prior)."""
    gp = SingleTaskGP(X_n1, y_n1)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()
    return gp


def warm_start_n2_gp(X_pool_norm, gp_n1):
    """
    Seed the N2 GP using N1 GP predictions over the full pool.
    This transfers N1 knowledge before any real N2 label is seen.
    """
    with torch.no_grad():
        y_seed = gp_n1.posterior(X_pool_norm).mean   # (N_pool, 1)

    gp = SingleTaskGP(X_pool_norm.clone(), y_seed.clone())
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


def rebuild_n2_gp(X_obs, y_obs):
    """Refit the N2 GP from scratch on accumulated observations."""
    gp = SingleTaskGP(X_obs, y_obs)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


def n1_threshold_std(y1_mean, y1_std):
    """Convert real-unit N1 threshold to standardized space."""
    return (TOY_N1_THRESHOLD_REAL - y1_mean) / y1_std


def get_feasible_mask(X_pool_norm, gp_n1, threshold_std):
    """
    Hard-mask: True where the N1 UCB is BELOW the threshold.
    Points above threshold are considered infeasible for N2 collection.
    """
    with torch.no_grad():
        post  = gp_n1.posterior(X_pool_norm)
        mean  = post.mean.squeeze(-1)
        std   = post.variance.sqrt().squeeze(-1)
    ucb_n1 = mean + CONSTRAINT_CONFIDENCE * std
    return ucb_n1 < threshold_std


# ══════════════════════════════════════════════════════════════════════════════
# METHOD A — HARD MASK
# ══════════════════════════════════════════════════════════════════════════════

def run_hard_mask(X_n1, y_n1, y1_mean, y1_std,
                  X_pool_norm, y_pool_std, y_pool_raw,
                  seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gp_n1     = fit_n1_gp(X_n1, y_n1)
    threshold = n1_threshold_std(y1_mean, y1_std)

    # Seed one real N2 observation so the GP starts from actual data, not N1 predictions
    init_idx = int(torch.randint(0, X_pool_norm.shape[0], (1,)).item())
    queried  = {init_idx}
    X_real   = X_pool_norm[init_idx].unsqueeze(0)
    y_real   = y_pool_std[init_idx].unsqueeze(0)
    gp_n2    = rebuild_n2_gp(X_real, y_real)

    best_values    = []
    selected_values = []
    n_feasible_log = []

    for i in range(N_ITERATIONS):
        gp_n2.eval()

        # Score all pool points
        with torch.no_grad():
            acq      = UpperConfidenceBound(gp_n2, beta=UCB_BETA)
            acq_vals = acq(X_pool_norm.unsqueeze(1))   # (N_pool,)

        # Apply hard mask
        feasible_mask = get_feasible_mask(X_pool_norm, gp_n1, threshold)
        queried_mask = torch.zeros(X_pool_norm.shape[0], dtype=torch.bool)
        for idx in queried:
            queried_mask[idx] = True
            acq_vals[idx] = -float('inf')
        acq_vals[~feasible_mask] = -float('inf')

        # Count remaining candidates: feasible AND not yet queried
        n_feasible = (feasible_mask & ~queried_mask).sum().item()
        n_feasible_log.append(n_feasible)

        if (acq_vals == -float('inf')).all():
            print(f"[Hard Mask] Iter {i+1}: No feasible points — stopping early.")
            last = best_values[-1] if best_values else 0.0
            best_values.extend([last] * (N_ITERATIONS - i))
            selected_values.extend([np.nan] * (N_ITERATIONS - i))
            n_feasible_log.extend([0] * (N_ITERATIONS - i))
            break

        best_idx = acq_vals.argmax().item()
        queried.add(best_idx)

        x_new = X_pool_norm[best_idx].unsqueeze(0)
        y_new = y_pool_std[best_idx].unsqueeze(0)

        X_real = torch.cat([X_real, x_new], dim=0)
        y_real = torch.cat([y_real, y_new], dim=0)

        gp_n2 = rebuild_n2_gp(X_real, y_real)

        best_raw     = y_pool_raw[list(queried)].max().item()
        selected_raw = y_pool_raw[best_idx].item()
        best_values.append(best_raw)
        selected_values.append(selected_raw)

        print(f"  [Hard Mask | seed {seed}] Iter {i+1:2d} | "
              f"idx: {best_idx:4d} | feasible: {n_feasible:4d} | "
              f"selected: {selected_raw:.4f} | best: {best_raw:.4f}")

    return best_values, selected_values, n_feasible_log


# ══════════════════════════════════════════════════════════════════════════════
# METHOD B — ConstrainedExpectedImprovement (CEI)
# ══════════════════════════════════════════════════════════════════════════════

def run_cei(X_n1, y_n1, y1_mean, y1_std,
            X_pool_norm, y_pool_std, y_pool_raw,
            seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gp_n1     = fit_n1_gp(X_n1, y_n1)
    threshold = n1_threshold_std(y1_mean, y1_std)

    init_idx = int(torch.randint(0, X_pool_norm.shape[0], (1,)).item())
    queried  = {init_idx}
    X_real   = X_pool_norm[init_idx].unsqueeze(0)
    y_real   = y_pool_std[init_idx].unsqueeze(0)
    gp_n2    = rebuild_n2_gp(X_real, y_real)

    best_values     = []
    selected_values = []
    pof_log         = []   # probability of feasibility at selected point

    for i in range(N_ITERATIONS):
        gp_n2.eval()
        gp_n1.eval()

        # Stack: output 0 = N2 (objective), output 1 = N1 (constraint)
        # ModelListGP requires independent models with matching train inputs;
        # we use separate GPs and pass them as a list.
        joint_model = ModelListGP(gp_n2, gp_n1)

        best_f = y_pool_std[list(queried)].max()

        cei = ConstrainedExpectedImprovement(
            model=joint_model,
            best_f=best_f,
            objective_index=0,                      # maximise N2 GP output
            constraints={1: (None, threshold)},     # N1 GP output < threshold
        )

        with torch.no_grad():
            candidates = X_pool_norm.unsqueeze(1)   # (N_pool, 1, d)
            acq_vals   = cei(candidates)             # (N_pool,)

        for idx in queried:
            acq_vals[idx] = -float('inf')

        if (acq_vals == -float('inf')).all():
            print(f"[CEI] Iter {i+1}: No unqueried points — stopping early.")
            last = best_values[-1] if best_values else 0.0
            best_values.extend([last] * (N_ITERATIONS - i))
            selected_values.extend([np.nan] * (N_ITERATIONS - i))
            pof_log.extend([0.0] * (N_ITERATIONS - i))
            break

        best_idx = acq_vals.argmax().item()
        queried.add(best_idx)

        # Log P(feasible) at selected point for diagnostics
        with torch.no_grad():
            post_n1 = gp_n1.posterior(X_pool_norm[best_idx].unsqueeze(0))
            mean_n1 = post_n1.mean.item()
            std_n1  = post_n1.variance.sqrt().item()
            pof = Normal(mean_n1, std_n1).cdf(torch.tensor(threshold)).item()
        pof_log.append(pof)

        x_new = X_pool_norm[best_idx].unsqueeze(0)
        y_new = y_pool_std[best_idx].unsqueeze(0)

        X_real = torch.cat([X_real, x_new], dim=0)
        y_real = torch.cat([y_real, y_new], dim=0)

        gp_n2 = rebuild_n2_gp(X_real, y_real)

        best_raw     = y_pool_raw[list(queried)].max().item()
        selected_raw = y_pool_raw[best_idx].item()
        best_values.append(best_raw)
        selected_values.append(selected_raw)

        print(f"  [CEI       | seed {seed}] Iter {i+1:2d} | "
              f"idx: {best_idx:4d} | P(feasible): {pof:.3f} | "
              f"selected: {selected_raw:.4f} | best: {best_raw:.4f}")

    return best_values, selected_values, pof_log


# ══════════════════════════════════════════════════════════════════════════════
# METHOD C — HARD MASK + EXPECTED IMPROVEMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_hard_mask_ei(X_n1, y_n1, y1_mean, y1_std,
                     X_pool_norm, y_pool_std, y_pool_raw,
                     seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gp_n1     = fit_n1_gp(X_n1, y_n1)
    threshold = n1_threshold_std(y1_mean, y1_std)

    init_idx = int(torch.randint(0, X_pool_norm.shape[0], (1,)).item())
    queried  = {init_idx}
    X_real   = X_pool_norm[init_idx].unsqueeze(0)
    y_real   = y_pool_std[init_idx].unsqueeze(0)
    gp_n2    = rebuild_n2_gp(X_real, y_real)

    best_values     = []
    selected_values = []
    n_feasible_log  = []

    for i in range(N_ITERATIONS):
        gp_n2.eval()

        best_f = y_pool_std[list(queried)].max()

        with torch.no_grad():
            acq      = ExpectedImprovement(gp_n2, best_f=best_f)
            acq_vals = acq(X_pool_norm.unsqueeze(1))   # (N_pool,)

        feasible_mask = get_feasible_mask(X_pool_norm, gp_n1, threshold)
        queried_mask  = torch.zeros(X_pool_norm.shape[0], dtype=torch.bool)
        for idx in queried:
            queried_mask[idx] = True
            acq_vals[idx] = -float('inf')
        acq_vals[~feasible_mask] = -float('inf')

        n_feasible = (feasible_mask & ~queried_mask).sum().item()
        n_feasible_log.append(n_feasible)

        if (acq_vals == -float('inf')).all():
            print(f"[HM+EI] Iter {i+1}: No feasible points — stopping early.")
            last = best_values[-1] if best_values else 0.0
            best_values.extend([last] * (N_ITERATIONS - i))
            selected_values.extend([np.nan] * (N_ITERATIONS - i))
            n_feasible_log.extend([0] * (N_ITERATIONS - i))
            break

        best_idx = acq_vals.argmax().item()
        queried.add(best_idx)

        x_new = X_pool_norm[best_idx].unsqueeze(0)
        y_new = y_pool_std[best_idx].unsqueeze(0)

        X_real = torch.cat([X_real, x_new], dim=0)
        y_real = torch.cat([y_real, y_new], dim=0)

        gp_n2 = rebuild_n2_gp(X_real, y_real)

        best_raw     = y_pool_raw[list(queried)].max().item()
        selected_raw = y_pool_raw[best_idx].item()
        best_values.append(best_raw)
        selected_values.append(selected_raw)

        print(f"  [HM+EI    | seed {seed}] Iter {i+1:2d} | "
              f"idx: {best_idx:4d} | feasible: {n_feasible:4d} | "
              f"selected: {selected_raw:.4f} | best: {best_raw:.4f}")

    return best_values, selected_values, n_feasible_log


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(hm_runs, cei_runs, hmei_runs,
                 hm_sel, cei_sel, hmei_sel,
                 hm_diag, cei_diag):
    iters = np.arange(1, N_ITERATIONS + 1)

    def pad(runs, fill_last=True):
        padded = []
        for r in runs:
            r = list(r)
            fill = r[-1] if (fill_last and r) else np.nan
            while len(r) < N_ITERATIONS:
                r.append(fill)
            padded.append(r[:N_ITERATIONS])
        return np.array(padded, dtype=float)

    hm_arr    = pad(hm_runs)
    cei_arr   = pad(cei_runs)
    hmei_arr  = pad(hmei_runs)
    hm_sel_arr    = pad(hm_sel,   fill_last=False)
    cei_sel_arr   = pad(cei_sel,  fill_last=False)
    hmei_sel_arr  = pad(hmei_sel, fill_last=False)
    hm_d_arr  = pad(hm_diag)
    cei_d_arr = pad(cei_diag)

    fig = plt.figure(figsize=(13, 12))
    gs  = gridspec.GridSpec(3, 2, hspace=0.42, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, :])   # convergence (running best) — full width
    ax2 = fig.add_subplot(gs[1, :])   # raw selected score — full width
    ax3 = fig.add_subplot(gs[2, 0])   # hard-mask: # feasible points
    ax4 = fig.add_subplot(gs[2, 1])   # CEI: P(feasible) at selection

    def plot_band(ax, arr, color, label):
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr, axis=0)
        ax.plot(iters, mean, color=color, marker='o', markersize=4, label=label)
        if N_SEEDS > 1:
            ax.fill_between(iters, mean - std, mean + std, alpha=0.18, color=color)

    # ── Running best ──────────────────────────────────────────────────────────
    plot_band(ax1, hm_arr,   '#E8593C', 'Hard Mask + UCB (A)')
    plot_band(ax1, cei_arr,  '#3B8BD4', 'CEI (B)')
    plot_band(ax1, hmei_arr, '#2CA02C', 'Hard Mask + EI (C)')
    ax1.axhline(y=TOY_N1_THRESHOLD_REAL, color='gray', linestyle='--',
                linewidth=1, alpha=0.6, label=f'N1 threshold ({TOY_N1_THRESHOLD_REAL})')
    ax1.set_xlabel('BO Iteration', fontsize=11)
    ax1.set_ylabel('Best value so far', fontsize=11)
    ax1.set_title('Convergence (running best)', fontsize=12, fontweight='500')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    # ── Raw selected score ────────────────────────────────────────────────────
    plot_band(ax2, hm_sel_arr,   '#E8593C', 'Hard Mask + UCB (A)')
    plot_band(ax2, cei_sel_arr,  '#3B8BD4', 'CEI (B)')
    plot_band(ax2, hmei_sel_arr, '#2CA02C', 'Hard Mask + EI (C)')
    ax2.axhline(y=TOY_N1_THRESHOLD_REAL, color='gray', linestyle='--',
                linewidth=1, alpha=0.6, label=f'N1 threshold ({TOY_N1_THRESHOLD_REAL})')
    ax2.set_xlabel('BO Iteration', fontsize=11)
    ax2.set_ylabel('Score of selected point', fontsize=11)
    ax2.set_title('Raw selected score per iteration', fontsize=12, fontweight='500')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    # ── Hard Mask: feasible pool size over iterations ─────────────────────────
    mean_f = hm_d_arr.mean(0)
    ax3.bar(iters, mean_f, color='#E8593C', alpha=0.7, width=0.6)
    ax3.set_xlabel('Iteration', fontsize=10)
    ax3.set_ylabel('# Remaining candidates', fontsize=10)
    ax3.set_title('Hard Mask — available pool (feasible & unqueried)', fontsize=11)
    ax3.grid(axis='y', alpha=0.25)
    ax3.set_xticks(iters)

    # ── CEI: P(feasible) at each selected point ───────────────────────────────
    mean_p = cei_d_arr.mean(0)
    ax4.plot(iters, mean_p, color='#3B8BD4', marker='s', markersize=4)
    if N_SEEDS > 1:
        ax4.fill_between(iters, cei_d_arr.min(0), cei_d_arr.max(0),
                         alpha=0.18, color='#3B8BD4')
    ax4.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6,
                label='P=0.5')
    ax4.set_ylim(0, 1.05)
    ax4.set_xlabel('Iteration', fontsize=10)
    ax4.set_ylabel('P(N1 constraint satisfied)', fontsize=10)
    ax4.set_title('CEI — constraint satisfaction probability', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.25)
    ax4.set_xticks(iters)

    fig.suptitle(
        f'Constrained BO Benchmark  |  toy (N1 layer={TOY_N1_LAYER}, N2 layer={TOY_N2_LAYER})'
        f'  |  N1 threshold = {TOY_N1_THRESHOLD_REAL}  |  '
        f'{N_SEEDS} seed(s)  |  {N_ITERATIONS} iterations',
        fontsize=11, y=1.01
    )

    out = 'benchmark_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(hm_runs, cei_runs, hmei_runs):
    def stats(runs, label):
        final = np.array([r[-1] for r in runs])
        auc   = np.array([np.trapezoid(r) for r in runs])
        print(f"  {label}")
        print(f"    Final best QW2   : {final.mean():.4f} ± {final.std():.4f}")
        print(f"    AUC (convergence): {auc.mean():.2f}  ± {auc.std():.2f}")
        print(f"    Best single run  : {final.max():.4f}")

    print("\n" + "═"*54)
    print("  BENCHMARK SUMMARY")
    print("═"*54)
    stats(hm_runs,   "Method A — Hard Mask + UCB")
    print()
    stats(cei_runs,  "Method B — CEI")
    print()
    stats(hmei_runs, "Method C — Hard Mask + EI")
    print("═"*54)

    wins = {'A': 0, 'B': 0, 'C': 0}
    for i in range(N_SEEDS):
        scores = {
            'A': hm_runs[i][-1],
            'B': cei_runs[i][-1],
            'C': hmei_runs[i][-1],
        }
        best = max(scores, key=scores.get)
        wins[best] += 1
    print(f"\n  Method A (HM+UCB) won {wins['A']}/{N_SEEDS} runs")
    print(f"  Method B (CEI)    won {wins['B']}/{N_SEEDS} runs")
    print(f"  Method C (HM+EI)  won {wins['C']}/{N_SEEDS} runs\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating toy data...")
    rng = np.random.default_rng(42)

    xy_n1    = rng.uniform(0, 3, size=(TOY_N1_SAMPLES, 2))
    z_n1     = np.array([toy_function(x, y, TOY_N1_LAYER, sigma=TOY_SIGMA, noise_scale=0.05)
                         for x, y in xy_n1])
    X_n1_raw = torch.tensor(xy_n1, dtype=torch.float64)
    y_n1_raw = torch.tensor(z_n1,  dtype=torch.float64).unsqueeze(-1)

    g        = np.linspace(0, 3, TOY_POOL_GRID)
    xx, yy   = np.meshgrid(g, g)
    xy_pool  = np.stack([xx.ravel(), yy.ravel()], axis=1)
    z_pool   = np.array([toy_function(x, y, TOY_N2_LAYER, sigma=TOY_SIGMA, noise_scale=0.0)
                         for x, y in xy_pool])
    X_pool_raw = torch.tensor(xy_pool, dtype=torch.float64)
    y_pool_raw = torch.tensor(z_pool,  dtype=torch.float64).unsqueeze(-1)

    X_n1      = normalize(X_n1_raw,   TOY_BOUNDS)
    X_pool_norm = normalize(X_pool_raw, TOY_BOUNDS)
    y1_mean, y1_std = y_n1_raw.mean(), y_n1_raw.std().clamp(min=1e-6)
    y_n1      = (y_n1_raw  - y1_mean) / y1_std
    y2_mean, y2_std = y_pool_raw.mean(), y_pool_raw.std().clamp(min=1e-6)
    y_pool_std = (y_pool_raw - y2_mean) / y2_std

    print(f"\nN1 training points : {X_n1.shape[0]}")
    print(f"N2 pool points     : {X_pool_norm.shape[0]}")
    print(f"N1 threshold (real units): {TOY_N1_THRESHOLD_REAL}")
    threshold_std_val = (TOY_N1_THRESHOLD_REAL - y1_mean) / y1_std
    print(f"N1 threshold (std  units): {threshold_std_val:.3f}")

    # ── Constraint feasibility diagnostic ─────────────────────────────────────
    gp_n1_diag = fit_n1_gp(X_n1, y_n1)
    with torch.no_grad():
        post_diag    = gp_n1_diag.posterior(X_pool_norm)
        n1_pred_real = post_diag.mean.squeeze(-1) * y1_std + y1_mean

    y2_vals       = y_pool_raw.squeeze(-1)
    feasible_mask = n1_pred_real < TOY_N1_THRESHOLD_REAL

    top_k   = 10
    top_idx = y2_vals.argsort(descending=True)[:top_k]
    print(f"\n── Top-{top_k} Toy N2 pool points and their N1 GP predictions ──")
    print(f"  {'Rank':<5} {'Toy N2':>8} {'N1 pred (real)':>16} {'Feasible?':>10}")
    for rank, idx in enumerate(top_idx):
        q   = y2_vals[idx].item()
        n1p = n1_pred_real[idx].item()
        feas = "YES" if n1p < TOY_N1_THRESHOLD_REAL else "NO  <-- blocked"
        print(f"  {rank+1:<5} {q:>8.4f} {n1p:>16.4f} {feas:>10}")

    n_feasible_total = feasible_mask.sum().item()
    print(f"\n  Feasible pool points (N1 pred < {TOY_N1_THRESHOLD_REAL}): "
          f"{n_feasible_total} / {len(y2_vals)}")
    if feasible_mask.any():
        print(f"  Max Toy N2 in feasible region: "
              f"{y2_vals[feasible_mask].max().item():.4f}")
    print(f"  Max Toy N2 in full pool      : {y2_vals.max().item():.4f}")
    print()
    # ─────────────────────────────────────────────────────────────────────────

    print(f"Running {N_SEEDS} seed(s) × {N_ITERATIONS} iterations\n")

    hm_runs,   hm_sel,   hm_diag   = [], [], []
    cei_runs,  cei_sel,  cei_diag  = [], [], []
    hmei_runs, hmei_sel, hmei_diag = [], [], []

    for seed in range(N_SEEDS):
        print(f"── Seed {seed} ──────────────────────────────────")
        print("  Method A: Hard Mask + UCB")
        bv_hm, sel_hm, diag_hm = run_hard_mask(
            X_n1, y_n1, y1_mean, y1_std,
            X_pool_norm, y_pool_std, y_pool_raw,
            seed=seed
        )
        hm_runs.append(bv_hm)
        hm_sel.append(sel_hm)
        hm_diag.append(diag_hm)

        print("  Method B: CEI")
        bv_cei, sel_cei, diag_cei = run_cei(
            X_n1, y_n1, y1_mean, y1_std,
            X_pool_norm, y_pool_std, y_pool_raw,
            seed=seed
        )
        cei_runs.append(bv_cei)
        cei_sel.append(sel_cei)
        cei_diag.append(diag_cei)

        print("  Method C: Hard Mask + EI")
        bv_hmei, sel_hmei, diag_hmei = run_hard_mask_ei(
            X_n1, y_n1, y1_mean, y1_std,
            X_pool_norm, y_pool_std, y_pool_raw,
            seed=seed
        )
        hmei_runs.append(bv_hmei)
        hmei_sel.append(sel_hmei)
        hmei_diag.append(diag_hmei)

    print_summary(hm_runs, cei_runs, hmei_runs)
    plot_results(hm_runs, cei_runs, hmei_runs,
                 hm_sel, cei_sel, hmei_sel,
                 hm_diag, cei_diag)
