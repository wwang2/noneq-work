import torch
import numpy as np

def jarzynski_estimate(works, beta=1.0):
    """
    Compute free energy estimate using Jarzynski Equality:
    exp(-beta * DeltaF) = <exp(-beta * W)>
    
    works: Tensor of shape (num_trajectories,)
    """
    if works.shape[0] == 0:
        return torch.tensor(0.0)
        
    num_samples = works.shape[0]
    scaled_works = -beta * works
    
    log_sum_exp = torch.logsumexp(scaled_works, dim=0)
    delta_f = -(1.0 / beta) * (log_sum_exp - np.log(num_samples))
    
    return delta_f

def jarzynski_bootstrap_error(works, beta=1.0, num_bootstraps=100):
    """
    Estimate the uncertainty of the Jarzynski estimate using bootstrapping.
    """
    num_samples = works.shape[0]
    if num_samples < 2:
        return 0.0
        
    bootstrap_estimates = []
    works_np = works.cpu().numpy()
    
    for _ in range(num_bootstraps):
        resampled_works = torch.from_numpy(np.random.choice(works_np, size=num_samples, replace=True))
        bootstrap_estimates.append(jarzynski_estimate(resampled_works, beta).item())
        
    return np.std(bootstrap_estimates)

def jarzynski_convergence(works, beta=1.0, compute_error=True):
    """
    Compute Jarzynski estimate and optionally uncertainty as a function of sample size.
    """
    num_samples = works.shape[0]
    estimates = []
    errors = []
    
    # We'll skip every few samples for speed if num_samples is large
    step = max(1, num_samples // 100)
    indices = list(range(1, num_samples + 1, step))
    if indices[-1] != num_samples:
        indices.append(num_samples)
        
    for i in indices:
        est = jarzynski_estimate(works[:i], beta).item()
        estimates.append(est)
        if compute_error:
            # Fewer bootstraps for smaller sample sizes to save time
            n_boot = 50 if i < 100 else 100
            errors.append(jarzynski_bootstrap_error(works[:i], beta, num_bootstraps=n_boot))
        else:
            errors.append(0.0)
            
    return np.array(indices), np.array(estimates), np.array(errors)

def crooks_intersection(works_forward, works_reverse, beta=1.0):
    """
    Estimate Delta F from the intersection of forward and reverse work distributions.
    """
    df_fwd = jarzynski_estimate(works_forward, beta)
    df_rev = -jarzynski_estimate(-works_reverse, beta)
    return (df_fwd + df_rev) / 2.0
