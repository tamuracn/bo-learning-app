import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib; matplotlib.use('Agg')
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy.interpolate import interp1d

from toy_model.toy_function import toy_function
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


def _build_kde_soft_constraint(x_donor_01, y_target_01, target_thr, window=0.15, grid_res=200):
    qa_grid = np.linspace(0, 1, grid_res)
    p_soft = []
    for qa_val in qa_grid:
        w = np.exp(-0.5 * ((x_donor_01 - qa_val) / window) ** 2)
        w /= w.sum() + 1e-12
        p_soft.append(float((w * (y_target_01 > target_thr)).sum()))
    return interp1d(qa_grid, p_soft, bounds_error=False, fill_value=(p_soft[0], p_soft[-1]))


def _gp_map(gp, mg, Xpr_np, queried):
    g = np.linspace(0, 1, mg)
    Xg = torch.tensor(np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2), dtype=torch.float64)
    with torch.no_grad():
        post = gp.posterior(Xg)
        mean = post.mean.squeeze(-1).reshape(mg, mg).numpy()
        std  = post.variance.squeeze(-1).clamp(min=0).sqrt().reshape(mg, mg).numpy()
    real = np.linspace(0, 3, mg).tolist()
    idx  = sorted(queried)
    return {
        'x': real, 'y': real,
        'mean': mean.tolist(), 'std': std.tolist(),
        'qx': Xpr_np[idx, 0].tolist(), 'qy': Xpr_np[idx, 1].tolist(),
    }


def run_experiment(config, on_event):
    donor_layer  = int(config.get('donor_layer', 1))
    target_layer = int(config.get('target_layer', 2))
    n_donor      = int(config.get('donor_samples', 60))
    pg           = int(config.get('pool_grid', 35))
    thr_real     = float(config.get('donor_threshold', 2.0))
    sigma        = float(config.get('sigma', 0.5))
    n_iter       = int(config.get('n_iterations', 30))
    n_seeds      = int(config.get('n_seeds', 3))
    beta         = float(config.get('ucb_beta', 2.0))
    conf         = float(config.get('constraint_confidence', 1.28))
    batch_size   = max(1, int(config.get('batch_size', 1)))
    mg           = 20

    bounds = torch.tensor([[0., 0.], [3., 3.]], dtype=torch.float64)

    rng      = np.random.default_rng(42)
    xy_donor = rng.uniform(0, 3, (n_donor, 2))
    z_donor  = np.array([toy_function(x, y, donor_layer, sigma=sigma, noise_scale=0.05) for x, y in xy_donor])
    X_donor_raw = torch.tensor(xy_donor, dtype=torch.float64)
    y_donor_raw = torch.tensor(z_donor, dtype=torch.float64).unsqueeze(-1)

    g    = np.linspace(0, 3, pg)
    xyp  = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    zp   = np.array([toy_function(x, y, target_layer, sigma=sigma, noise_scale=0.0) for x, y in xyp])
    Xpr  = torch.tensor(xyp, dtype=torch.float64)
    ypr  = torch.tensor(zp, dtype=torch.float64).unsqueeze(-1)
    Np   = Xpr.shape[0]
    Xpr_np = xyp

    X_donor  = normalize(X_donor_raw, bounds)
    Xp       = normalize(Xpr, bounds)
    m_donor, s_donor = y_donor_raw.mean(), y_donor_raw.std().clamp(min=1e-6)
    y_donor  = (y_donor_raw - m_donor) / s_donor
    m_target, s_target = ypr.mean(), ypr.std().clamp(min=1e-6)
    yp       = (ypr - m_target) / s_target
    thr      = float((thr_real - m_donor) / s_donor)

    # Donor objective values at pool points (used by Method D donor BO and Method E KDE)
    zp_donor      = np.array([toy_function(x, y, donor_layer, sigma=sigma, noise_scale=0.0) for x, y in xyp])
    yp_donor_pool = (torch.tensor(zp_donor, dtype=torch.float64).unsqueeze(-1) - m_donor) / s_donor

    # KDE for Method E — min-max scale paired (donor, target) pool values to [0, 1]
    y_d_min, y_d_max = zp_donor.min(), zp_donor.max()
    y_t_min, y_t_max = zp.min(), zp.max()
    x_donor_01  = (zp_donor - y_d_min) / (y_d_max - y_d_min + 1e-12)
    y_target_01 = (zp       - y_t_min) / (y_t_max - y_t_min + 1e-12)
    kde_pct   = float(np.mean(zp_donor <= thr_real))
    n2_thr_01 = float(np.percentile(y_target_01, kde_pct * 100))
    kde_pf    = _build_kde_soft_constraint(x_donor_01, y_target_01, n2_thr_01)

    def fmask(gp_donor):
        with torch.no_grad():
            post = gp_donor.posterior(Xp)
            ucb  = post.mean.squeeze(-1) + conf * post.variance.sqrt().squeeze(-1)
        return ucb < thr

    def send(method, seed, it, sel_idx, queried, qmask, gp_target, extra=None):
        best = float(ypr[sorted(queried)].max())
        sel  = float(ypr[sel_idx])
        ev   = {'method': method, 'seed': seed, 'iter': it,
                'best': round(best, 6), 'sel': round(sel, 6)}
        if extra:
            ev.update(extra)
        ev['gp_map'] = _gp_map(gp_target, mg, Xpr_np, queried)
        on_event(ev)

    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        gp_donor = _fit_gp(X_donor, y_donor)
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

        # ── D: Data Transfer Warm Start + UCB ────────────────────────────────
        # Run donor BO on the pool for n_donor steps using the donor objective,
        # then re-label those pool points with target values as a warm start.
        aqui_func = "EI" # or "UCB"
        q_d = {init}
        qm_d = torch.zeros(Np, dtype=torch.bool); qm_d[init] = True
        Xo_d = Xp[init].unsqueeze(0)
        yo_d = yp_donor_pool[init].unsqueeze(0)
        gp_d = _fit_gp(Xo_d, yo_d)
        for _ in range(n_donor - 1):
            gp_d.eval()
            with torch.no_grad():
                if aqui_func == "UCB":
                    v_d = UpperConfidenceBound(gp_d, beta=beta)(Xp.unsqueeze(1))
                else:
                    bf_d = yp_donor_pool[sorted(q_d)].max()
                    v_d = ExpectedImprovement(gp_d, best_f=bf_d)(Xp.unsqueeze(1))
            v_d[qm_d] = -float('inf')
            if (v_d == -float('inf')).all(): break
            idx = int(v_d.argmax())
            q_d.add(idx); qm_d[idx] = True
            Xo_d = torch.cat([Xo_d, Xp[idx].unsqueeze(0)])
            yo_d = torch.cat([yo_d, yp_donor_pool[idx].unsqueeze(0)])
            gp_d = _fit_gp(Xo_d, yo_d)

        # Warm-start target GP with target values at donor-selected pool points
        q = set(q_d)
        qm = torch.zeros(Np, dtype=torch.bool)
        for i in q:
            qm[i] = True
        Xo = Xp[sorted(q)]
        yo = yp[sorted(q)]
        gp_target = _fit_gp(Xo, yo)
        for it in range(1, n_iter + 1):
            gp_target.eval()
            with torch.no_grad():
                if aqui_func == "UCB":
                    v_d = UpperConfidenceBound(gp_d, beta=beta)(Xp.unsqueeze(1))
                else:
                    bf_d = yp_donor_pool[sorted(q_d)].max()
                    v_d = ExpectedImprovement(gp_d, best_f=bf_d)(Xp.unsqueeze(1))
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
            send('D', seed, it, sel, q, qm, gp_target, {'n_transfer': len(q_d)})

        # ── E: KDE Soft-Constraint EI ─────────────────────────────────────────
        # Weight EI by empirical P(target > threshold | donor = x) from the KDE.
        q = {init}
        qm = torch.zeros(Np, dtype=torch.bool); qm[init] = True
        Xo = Xp[init].unsqueeze(0); yo = yp[init].unsqueeze(0)
        gp_target = _fit_gp(Xo, yo)
        for it in range(1, n_iter + 1):
            gp_target.eval(); gp_donor.eval()
            bf = yp[sorted(q)].max()
            with torch.no_grad():
                ei_vals      = ExpectedImprovement(gp_target, best_f=bf)(Xp.unsqueeze(1))
                donor_mu_std = gp_donor.posterior(Xp).mean.squeeze(-1)
            donor_mu_raw = donor_mu_std * float(s_donor) + float(m_donor)
            donor_mu_01  = ((donor_mu_raw.numpy() - y_d_min) / (y_d_max - y_d_min + 1e-12)).clip(0, 1)
            p_feas = torch.tensor(kde_pf(donor_mu_01), dtype=torch.float64)
            v = ei_vals * p_feas
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
            send('E', seed, it, sel, q, qm, gp_target, {'pof': round(float(p_feas[sel]), 4)})
