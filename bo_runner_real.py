import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib; matplotlib.use('Agg')
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_model.imod_oracle import load_pool
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import (
    UpperConfidenceBound, ConstrainedExpectedImprovement, ExpectedImprovement
)
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.distributions import Normal


def _fit_gp(X, y):
    gp = SingleTaskGP(X, y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp.eval()


def _gp_map(gp, X_pool_np, bounds, queried, mg=20):
    """
    2D slice of the 4D GP posterior over R MAI × R BAAc,
    fixing Anneal Time and Temperature at their pool means.
    Returns the same dict format as bo_runner._gp_map so the frontend works unchanged.
    """
    anneal_mean = X_pool_np[:, 0].mean()
    temp_mean   = X_pool_np[:, 1].mean()
    r_mai_vals  = np.linspace(X_pool_np[:, 2].min(), X_pool_np[:, 2].max(), mg)
    r_baac_vals = np.linspace(X_pool_np[:, 3].min(), X_pool_np[:, 3].max(), mg)
    R_BAAc_grid, R_MAI_grid = np.meshgrid(r_baac_vals, r_mai_vals)

    X_grid_np = np.column_stack([
        np.full(R_MAI_grid.size, anneal_mean),
        np.full(R_MAI_grid.size, temp_mean),
        R_MAI_grid.ravel(),
        R_BAAc_grid.ravel(),
    ])
    X_grid = normalize(torch.tensor(X_grid_np, dtype=torch.float64), bounds)
    with torch.no_grad():
        post = gp.posterior(X_grid)
        mean = post.mean.squeeze(-1).reshape(mg, mg).numpy()
        std  = post.variance.squeeze(-1).clamp(min=0).sqrt().reshape(mg, mg).numpy()

    idx = sorted(queried)
    return {
        'x': r_baac_vals.tolist(), 'y': r_mai_vals.tolist(),
        'mean': mean.tolist(), 'std': std.tolist(),
        'qx': X_pool_np[idx, 3].tolist(),   # R BAAc of queried points
        'qy': X_pool_np[idx, 2].tolist(),   # R MAI of queried points
    }


def run_experiment_real(config, on_event):
    donor_qw    = config.get('donor_qw', 'QW1')
    target_qw   = config.get('target_qw', 'QW99')
    csv_path    = config.get('csv_path', None)
    thr_real    = float(config.get('donor_threshold', 0.0))
    n_iter      = int(config.get('n_iterations', 30))
    n_seeds     = int(config.get('n_seeds', 3))
    beta        = float(config.get('ucb_beta', 2.0))
    conf        = float(config.get('constraint_confidence', 1.28))
    batch_size  = max(1, int(config.get('batch_size', 1)))
    donor_max_pts = int(config.get('donor_max_pts', 300))

    X_pool_np, y_donor_pool_np, y_target_pool_np, X_donor_np, y_donor_train_np = load_pool(
        csv_path, donor_qw, target_qw
    )
    Np = len(X_pool_np)
    if Np == 0:
        raise ValueError(f"No pool rows have both {donor_qw} and {target_qw}. Check QW column names.")

    # Bounds from union of donor training and pool data
    X_all  = np.vstack([X_pool_np, X_donor_np])
    bounds = torch.tensor(
        np.vstack([X_all.min(0), X_all.max(0)]),
        dtype=torch.float64
    )

    X_donor_raw  = torch.tensor(X_donor_np, dtype=torch.float64)
    y_donor_raw  = torch.tensor(y_donor_train_np, dtype=torch.float64).unsqueeze(-1)
    Xpr          = torch.tensor(X_pool_np, dtype=torch.float64)
    ypr          = torch.tensor(y_target_pool_np, dtype=torch.float64).unsqueeze(-1)

    X_donor  = normalize(X_donor_raw, bounds)
    Xp       = normalize(Xpr, bounds)

    m_donor, s_donor = y_donor_raw.mean(), y_donor_raw.std().clamp(min=1e-6)
    y_donor  = (y_donor_raw - m_donor) / s_donor

    m_target, s_target = ypr.mean(), ypr.std().clamp(min=1e-6)
    yp       = (ypr - m_target) / s_target

    thr = float((thr_real - m_donor) / s_donor)

    def fmask(gp_donor):
        with torch.no_grad():
            post = gp_donor.posterior(Xp)
            ucb  = post.mean.squeeze(-1) + conf * post.variance.sqrt().squeeze(-1)
        return ucb < thr

    def send(method, seed, it, sel_idx, queried, qm, gp_target, extra=None):
        best = float(ypr[sorted(queried)].max())
        sel  = float(ypr[sel_idx])
        ev   = {'method': method, 'seed': seed, 'iter': it,
                'best': round(best, 6), 'sel': round(sel, 6)}
        if extra:
            ev.update(extra)
        ev['gp_map'] = _gp_map(gp_target, X_pool_np, bounds, queried)
        on_event(ev)

    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)

        # Subsample donor training data for GP tractability
        n_donor_use = min(len(X_donor_np), donor_max_pts)
        idx_donor   = np.random.choice(len(X_donor_np), n_donor_use, replace=False)
        gp_donor = _fit_gp(X_donor[idx_donor], y_donor[idx_donor])
        fm       = fmask(gp_donor)
        init     = int(torch.randint(0, Np, (1,)).item())

        # ── A: Hard Mask + UCB ────────────────────────────────────────────────
        q = {init}
        qm = torch.zeros(Np, dtype=torch.bool); qm[init] = True
        Xo = Xp[init].unsqueeze(0); yo = yp[init].unsqueeze(0)
        gp_target = _fit_gp(Xo, yo)
        for it in range(1, n_iter + 1):
            gp_target.eval()
            with torch.no_grad():
                v = UpperConfidenceBound(gp_target, beta=beta)(Xp.unsqueeze(1))
            v[qm] = -float('inf'); v[~fm] = -float('inf')
            if (v == -float('inf')).all(): break
            batch = []
            v_b = v.clone()
            for _ in range(batch_size):
                if (v_b == -float('inf')).all(): break
                idx = int(v_b.argmax()); batch.append(idx); v_b[idx] = -float('inf')
            for idx in batch:
                q.add(idx); qm[idx] = True
                Xo = torch.cat([Xo, Xp[idx].unsqueeze(0)])
                yo = torch.cat([yo, yp[idx].unsqueeze(0)])
            gp_target = _fit_gp(Xo, yo)
            nf  = int((fm & ~qm).sum())
            sel = max(batch, key=lambda i: float(ypr[i]))
            send('A', seed, it, sel, q, qm, gp_target, {'n_feasible': nf})

        # ── B: CEI ───────────────────────────────────────────────────────────
        q = {init}
        qm = torch.zeros(Np, dtype=torch.bool); qm[init] = True
        Xo = Xp[init].unsqueeze(0); yo = yp[init].unsqueeze(0)
        gp_target = _fit_gp(Xo, yo)
        for it in range(1, n_iter + 1):
            gp_target.eval(); gp_donor.eval()
            bf  = yp[sorted(q)].max()
            cei = ConstrainedExpectedImprovement(
                ModelListGP(gp_target, gp_donor), best_f=bf,
                objective_index=0, constraints={1: (None, thr)}
            )
            with torch.no_grad():
                v = cei(Xp.unsqueeze(1))
            v[qm] = -float('inf')
            if (v == -float('inf')).all(): break
            batch = []
            v_b = v.clone()
            for _ in range(batch_size):
                if (v_b == -float('inf')).all(): break
                idx = int(v_b.argmax()); batch.append(idx); v_b[idx] = -float('inf')
            for idx in batch:
                q.add(idx); qm[idx] = True
                Xo = torch.cat([Xo, Xp[idx].unsqueeze(0)])
                yo = torch.cat([yo, yp[idx].unsqueeze(0)])
            gp_target = _fit_gp(Xo, yo)
            sel = max(batch, key=lambda i: float(ypr[i]))
            with torch.no_grad():
                p_donor = gp_donor.posterior(Xp[sel].unsqueeze(0))
                pof = float(Normal(p_donor.mean.squeeze(), p_donor.variance.sqrt().squeeze())
                            .cdf(torch.tensor(thr, dtype=torch.float64)))
            send('B', seed, it, sel, q, qm, gp_target, {'pof': round(pof, 4)})

        # ── C: Hard Mask + EI ────────────────────────────────────────────────
        q = {init}
        qm = torch.zeros(Np, dtype=torch.bool); qm[init] = True
        Xo = Xp[init].unsqueeze(0); yo = yp[init].unsqueeze(0)
        gp_target = _fit_gp(Xo, yo)
        for it in range(1, n_iter + 1):
            gp_target.eval()
            bf = yp[sorted(q)].max()
            with torch.no_grad():
                v = ExpectedImprovement(gp_target, best_f=bf)(Xp.unsqueeze(1))
            v[qm] = -float('inf'); v[~fm] = -float('inf')
            if (v == -float('inf')).all(): break
            batch = []
            v_b = v.clone()
            for _ in range(batch_size):
                if (v_b == -float('inf')).all(): break
                idx = int(v_b.argmax()); batch.append(idx); v_b[idx] = -float('inf')
            for idx in batch:
                q.add(idx); qm[idx] = True
                Xo = torch.cat([Xo, Xp[idx].unsqueeze(0)])
                yo = torch.cat([yo, yp[idx].unsqueeze(0)])
            gp_target = _fit_gp(Xo, yo)
            nf  = int((fm & ~qm).sum())
            sel = max(batch, key=lambda i: float(ypr[i]))
            send('C', seed, it, sel, q, qm, gp_target, {'n_feasible': nf})
