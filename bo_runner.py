import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib; matplotlib.use('Agg')
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
    n1_layer = int(config.get('n1_layer', 1))
    n2_layer = int(config.get('n2_layer', 2))
    n1_n     = int(config.get('n1_samples', 60))
    pg       = int(config.get('pool_grid', 35))
    thr_real = float(config.get('n1_threshold', 2.0))
    sigma    = float(config.get('sigma', 0.5))
    n_iter   = int(config.get('n_iterations', 30))
    n_seeds  = int(config.get('n_seeds', 3))
    beta     = float(config.get('ucb_beta', 2.0))
    conf     = float(config.get('constraint_confidence', 1.28))
    mg       = 20

    bounds = torch.tensor([[0., 0.], [3., 3.]], dtype=torch.float64)

    rng = np.random.default_rng(42)
    xy1 = rng.uniform(0, 3, (n1_n, 2))
    z1  = np.array([toy_function(x, y, n1_layer, sigma=sigma, noise_scale=0.05) for x, y in xy1])
    Xn1r = torch.tensor(xy1, dtype=torch.float64)
    yn1r = torch.tensor(z1, dtype=torch.float64).unsqueeze(-1)

    g   = np.linspace(0, 3, pg)
    xyp = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    zp  = np.array([toy_function(x, y, n2_layer, sigma=sigma, noise_scale=0.0) for x, y in xyp])
    Xpr  = torch.tensor(xyp, dtype=torch.float64)
    ypr  = torch.tensor(zp, dtype=torch.float64).unsqueeze(-1)
    Np   = Xpr.shape[0]
    Xpr_np = xyp

    Xn1 = normalize(Xn1r, bounds)
    Xp  = normalize(Xpr, bounds)
    m1, s1 = yn1r.mean(), yn1r.std().clamp(min=1e-6)
    yn1 = (yn1r - m1) / s1
    m2, s2 = ypr.mean(), ypr.std().clamp(min=1e-6)
    yp  = (ypr - m2) / s2
    thr = float((thr_real - m1) / s1)

    def fmask(gn1):
        with torch.no_grad():
            post = gn1.posterior(Xp)
            ucb  = post.mean.squeeze(-1) + conf * post.variance.sqrt().squeeze(-1)
        return ucb < thr

    def send(method, seed, it, sel_idx, queried, qmask, gn2, extra=None):
        best = float(ypr[sorted(queried)].max())
        sel  = float(ypr[sel_idx])
        ev   = {'method': method, 'seed': seed, 'iter': it,
                'best': round(best, 6), 'sel': round(sel, 6)}
        if extra:
            ev.update(extra)
        if seed == 0:
            ev['gp_map'] = _gp_map(gn2, mg, Xpr_np, queried)
        on_event(ev)

    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        gn1  = _fit_gp(Xn1, yn1)
        fm   = fmask(gn1)
        init = int(torch.randint(0, Np, (1,)).item())

        # ── A: Hard Mask + UCB ────────────────────────────────────────────────
        q = {init}
        qm = torch.zeros(Np, dtype=torch.bool); qm[init] = True
        Xo = Xp[init].unsqueeze(0); yo = yp[init].unsqueeze(0)
        gn2 = _fit_gp(Xo, yo)
        for it in range(1, n_iter + 1):
            gn2.eval()
            with torch.no_grad():
                v = UpperConfidenceBound(gn2, beta=beta)(Xp.unsqueeze(1))
            v[qm] = -float('inf'); v[~fm] = -float('inf')
            if (v == -float('inf')).all(): break
            idx = int(v.argmax())
            q.add(idx); qm[idx] = True
            Xo = torch.cat([Xo, Xp[idx].unsqueeze(0)])
            yo = torch.cat([yo, yp[idx].unsqueeze(0)])
            gn2 = _fit_gp(Xo, yo)
            nf  = int((fm & ~qm).sum())
            send('A', seed, it, idx, q, qm, gn2, {'n_feasible': nf})

        # ── B: CEI ───────────────────────────────────────────────────────────
        q = {init}
        qm = torch.zeros(Np, dtype=torch.bool); qm[init] = True
        Xo = Xp[init].unsqueeze(0); yo = yp[init].unsqueeze(0)
        gn2 = _fit_gp(Xo, yo)
        for it in range(1, n_iter + 1):
            gn2.eval(); gn1.eval()
            bf  = yp[sorted(q)].max()
            cei = ConstrainedExpectedImprovement(
                ModelListGP(gn2, gn1), best_f=bf,
                objective_index=0, constraints={1: (None, thr)}
            )
            with torch.no_grad():
                v = cei(Xp.unsqueeze(1))
            v[qm] = -float('inf')
            if (v == -float('inf')).all(): break
            idx = int(v.argmax())
            q.add(idx); qm[idx] = True
            Xo = torch.cat([Xo, Xp[idx].unsqueeze(0)])
            yo = torch.cat([yo, yp[idx].unsqueeze(0)])
            gn2 = _fit_gp(Xo, yo)
            with torch.no_grad():
                pn1 = gn1.posterior(Xp[idx].unsqueeze(0))
                pof = float(Normal(pn1.mean.squeeze(), pn1.variance.sqrt().squeeze())
                            .cdf(torch.tensor(thr, dtype=torch.float64)))
            send('B', seed, it, idx, q, qm, gn2, {'pof': round(pof, 4)})

        # ── C: Hard Mask + EI ────────────────────────────────────────────────
        q = {init}
        qm = torch.zeros(Np, dtype=torch.bool); qm[init] = True
        Xo = Xp[init].unsqueeze(0); yo = yp[init].unsqueeze(0)
        gn2 = _fit_gp(Xo, yo)
        for it in range(1, n_iter + 1):
            gn2.eval()
            bf = yp[sorted(q)].max()
            with torch.no_grad():
                v = ExpectedImprovement(gn2, best_f=bf)(Xp.unsqueeze(1))
            v[qm] = -float('inf'); v[~fm] = -float('inf')
            if (v == -float('inf')).all(): break
            idx = int(v.argmax())
            q.add(idx); qm[idx] = True
            Xo = torch.cat([Xo, Xp[idx].unsqueeze(0)])
            yo = torch.cat([yo, yp[idx].unsqueeze(0)])
            gn2 = _fit_gp(Xo, yo)
            nf  = int((fm & ~qm).sum())
            send('C', seed, it, idx, q, qm, gn2, {'n_feasible': nf})
