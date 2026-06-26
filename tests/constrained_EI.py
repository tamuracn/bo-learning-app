from torch.distributions import Normal
import torch

def custom_cei(x, gp_n2, gp_n1, best_f, target_n1, tolerance):
    """
    x:          candidate points (n x 1 x d)
    gp_n2:      GP over N2 (objective)
    gp_n1:      GP over N1 (constraint, frozen)
    best_f:     best N2 value seen so far (standardised)
    target_n1:  the N1 target value (standardised)
    tolerance:  how close to target counts as feasible (standardised)
    """
    # --- EI on N2 ---
    post_n2 = gp_n2.posterior(x)
    mu_n2   = post_n2.mean.squeeze(-1)
    sig_n2  = post_n2.variance.squeeze(-1).clamp(min=1e-10).sqrt()

    z     = (mu_n2 - best_f) / sig_n2
    std_n = Normal(torch.zeros_like(z), torch.ones_like(z))
    ei    = sig_n2 * (std_n.log_prob(z).exp() + z * std_n.cdf(z))

    # --- Soft feasibility from N1 GP ---
    post_n1 = gp_n1.posterior(x)
    mu_n1   = post_n1.mean.squeeze(-1)
    sig_n1  = post_n1.variance.squeeze(-1).clamp(min=1e-10).sqrt()

    normal  = Normal(torch.zeros_like(mu_n1), torch.ones_like(sig_n1))
    phi     = normal.cdf((target_n1 + tolerance - mu_n1) / sig_n1) \
            - normal.cdf((target_n1 - tolerance - mu_n1) / sig_n1)

    return ei * phi