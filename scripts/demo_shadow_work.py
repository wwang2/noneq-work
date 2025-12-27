import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from noneq.potentials import DoubleWellPotential
from noneq.integrators import OverdampedLangevin

# Applying plotting style
plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["DejaVu Sans Mono", "Menlo", "Consolas", "Monaco"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# --- Helper Classes (MALA, RWM) ---
class MALAIntegrator:
    """Metropolis-Adjusted Langevin Algorithm"""
    def __init__(self, potential, gamma=1.0, kT=1.0, dt=0.01):
        self.potential = potential
        self.gamma = gamma
        self.kT = kT
        self.dt = dt
        self.beta = 1.0 / kT
    
    def step(self, x, lmbda):
        f_old = self.potential.force(x, lmbda)
        u_old = self.potential(x, lmbda)
        
        if torch.isnan(f_old).any() or torch.abs(f_old).max() > 1e4:
             return x, torch.zeros_like(x, dtype=torch.bool)

        noise = torch.randn_like(x) * np.sqrt(2 * self.kT * self.dt / self.gamma)
        x_new = x + (f_old / self.gamma) * self.dt + noise
        
        f_new = self.potential.force(x_new, lmbda)
        u_new = self.potential(x_new, lmbda)
        
        mu_fwd = x + (f_old / self.gamma) * self.dt
        log_q_fwd = - (x_new - mu_fwd)**2 / (4 * self.kT * self.dt / self.gamma)
        
        mu_rev = x_new + (f_new / self.gamma) * self.dt
        log_q_rev = - (x - mu_rev)**2 / (4 * self.kT * self.dt / self.gamma)
        
        log_pi_ratio = -self.beta * (u_new - u_old)
        log_ratio = log_pi_ratio + (log_q_rev - log_q_fwd)
        
        valid = torch.isfinite(log_ratio)
        log_ratio[~valid] = -float('inf') 
        
        accept_prob = torch.exp(torch.clamp(log_ratio, max=0.0))
        accepted = torch.rand_like(accept_prob) < accept_prob
        final_x = torch.where(accepted, x_new, x)
        
        return final_x, accepted

    def run(self, x_init, num_steps):
        x = x_init.clone()
        trajectories = [x.clone()]
        accept_counts = torch.zeros_like(x)
        
        for _ in range(num_steps):
            x, accepted = self.step(x, 0.0)
            accept_counts += accepted.float()
            trajectories.append(x.clone())
            
        return torch.stack(trajectories), accept_counts / num_steps

class UnderdampedLangevin:
    """
    Underdamped Langevin using BAOAB splitting.
    Tracks Shadow Work via O-step transition probability ratio.
    """
    def __init__(self, potential, gamma=1.0, kT=1.0, dt=0.01, mass=1.0):
        self.potential = potential
        self.gamma = gamma
        self.kT = kT
        self.dt = dt
        self.beta = 1.0 / kT
        self.mass = mass
        
        # Constants for O-step (Ornstein-Uhlenbeck)
        self.alpha = np.exp(-gamma * dt)
        self.sigma_v = np.sqrt(kT / mass * (1 - self.alpha**2))
        
    def step(self, x, v, lmbda):
        # Initial Energy
        u_old = self.potential(x, lmbda)
        ke_old = 0.5 * self.mass * v**2
        h_old = u_old + ke_old
        
        # B: Velocity half-step
        force = self.potential.force(x, lmbda)
        force = torch.clamp(force, -20.0, 20.0) # Safety clip for extreme dt
        v = v + (self.dt / 2) * force / self.mass
        
        # A: Position half-step
        x = x + (self.dt / 2) * v
        
        # O: Noise step on velocity
        v_before_o = v
        noise = torch.randn_like(v) * self.sigma_v
        v = self.alpha * v + noise
        v_after_o = v
        
        # Log Ratio of O-step probabilities
        term_fwd = (v_after_o - self.alpha * v_before_o)**2
        term_rev = (v_before_o - self.alpha * v_after_o)**2
        log_prob_ratio = (1.0 / (2.0 * self.sigma_v**2)) * (term_rev - term_fwd)
        
        # A: Position half-step
        x = x + (self.dt / 2) * v
        
        # B: Velocity half-step
        force = self.potential.force(x, lmbda) 
        force = torch.clamp(force, -20.0, 20.0)
        v = v + (self.dt / 2) * force / self.mass
        
        # Final Energy
        u_new = self.potential(x, lmbda)
        ke_new = 0.5 * self.mass * v**2
        h_new = u_new + ke_new
        
        # Shadow Work
        beta_w_shadow = self.beta * (h_new - h_old) + log_prob_ratio
        work_shadow = beta_w_shadow * self.kT
        
        return x, v, work_shadow
        
    def run_protocol(self, x_init, v_init, lambdas):
        num_steps = len(lambdas)
        x = x_init.clone()
        v = v_init.clone()
        
        trajectories = [x.clone()]
        total_shadow_work = torch.zeros_like(x)
        shadow_works = [total_shadow_work.clone()]
        
        for i in range(num_steps - 1):
            lmbda = lambdas[i]
            x, v, dw_s = self.step(x, v, lmbda)
            
            total_shadow_work += dw_s
            trajectories.append(x.clone())
            shadow_works.append(total_shadow_work.clone())
            
        return torch.stack(trajectories), torch.stack(shadow_works)

class RWMIntegrator:
    """Random Walk Metropolis"""
    def __init__(self, potential, gamma=1.0, kT=1.0, dt=0.01):
        self.potential = potential
        self.kT = kT
        self.beta = 1.0 / kT
        self.sigma = np.sqrt(2 * kT * dt / gamma)
    
    def step(self, x, lmbda):
        u_old = self.potential(x, lmbda)
        x_new = x + torch.randn_like(x) * self.sigma
        u_new = self.potential(x_new, lmbda)
        log_ratio = -self.beta * (u_new - u_old)
        accept_prob = torch.exp(torch.clamp(log_ratio, max=0.0))
        accepted = torch.rand_like(accept_prob) < accept_prob
        final_x = torch.where(accepted, x_new, x)
        return final_x, accepted

    def run(self, x_init, num_steps):
        x = x_init.clone()
        trajectories = [x.clone()]
        accept_counts = torch.zeros_like(x)
        for _ in range(num_steps):
            x, accepted = self.step(x, 0.0)
            accept_counts += accepted.float()
            trajectories.append(x.clone())
        return torch.stack(trajectories), accept_counts / num_steps

def calculate_ess_dynamic(traj):
    """ESS based on autocorrelation"""
    traj = traj.numpy()
    n_steps, n_chains = traj.shape
    ess_values = []
    
    for i in range(n_chains):
        chain = traj[:, i]
        chain = chain - np.mean(chain)
        c0 = np.var(chain)
        if c0 == 0:
            ess_values.append(1.0)
            continue
            
        acf = np.correlate(chain, chain, mode='full')[n_steps-1:] / (c0 * n_steps)
        cutoff = np.where(acf < 0.05)[0]
        limit = cutoff[0] if len(cutoff) > 0 else len(acf) // 10
        tau = 1 + 2 * np.sum(acf[1:limit])
        ess = n_steps / tau
        ess_values.append(ess)
        
    return np.mean(ess_values)

def get_boltzmann_samples(potential, lmbda, kT, num_samples, x_range=(-2.5, 2.5)):
    """Sample from exact Boltzmann distribution using rejection sampling."""
    xs = torch.linspace(x_range[0], x_range[1], 1000)
    us = potential(xs, lmbda)
    weights = torch.exp(-us / kT)
    max_w = weights.max()
    
    samples = []
    while len(samples) < num_samples:
        x_try = torch.rand(num_samples) * (x_range[1] - x_range[0]) + x_range[0]
        u_try = potential(x_try, lmbda)
        w_try = torch.exp(-u_try / kT)
        accept = torch.rand(num_samples) < (w_try / max_w)
        samples.extend(x_try[accept].tolist())
    
    return torch.tensor(samples[:num_samples])

def run_dashboard_demo():
    # Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Parameters
    kT = 1.0
    beta = 1.0 / kT
    gamma = 1.0
    num_trajectories = 80000 
    T_total_hist = 4.0   # For Histograms
    T_total_eff = 50.0   # For Efficiency Sweep
    
    # Potential (move to device later if needed, but it's just a class)
    potential = DoubleWellPotential(a=1.0, b=2.0)
    
    # Plotting Setup
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.4, wspace=0.25)
    
    ax_hist_shadow = fig.add_subplot(gs[0, 0])
    ax_hist_mala = fig.add_subplot(gs[0, 1])
    ax_eff = fig.add_subplot(gs[1, 0])
    ax_acc = fig.add_subplot(gs[1, 1])
    ax_accum = fig.add_subplot(gs[2, :]) # Full width for accumulation
    
    assets_dir = "/Users/wujiewang/projects/noneq-work/assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    # Colors (Consistent Theme)
    c_true = 'black'
    c_over = '#ff7f0e' # Orange
    c_baoab = '#9467bd' # Purple
    c_mala = '#1f77b4' # Blue
    c_rwm = '#7f7f7f' # Gray
    
    # --- PART 1: DISTRIBUTIONS (Shadow vs MALA) ---
    print("\n--- Generating Distributions (Shadow vs MALA) ---")
    
    # Pre-calculate true distribution
    x_plot = torch.linspace(-2.5, 2.5, 200).to(device)
    u_plot = potential(x_plot, 0.0)
    p_true = torch.exp(-u_plot / kT)
    p_true /= p_true.sum() * (x_plot[1] - x_plot[0])
    
    # Target dt for visualization
    dt_viz_over = 0.05
    dt_viz_baoab = 0.5 # Requested large step for debiasing
    
    x_init = get_boltzmann_samples(potential, 0.0, kT, num_trajectories, x_range=(-2.5, 2.5)).to(device)
    
    # 1a. Shadow Work Simulation (Overdamped)
    print(f"Running Overdamped Shadow Work Simulation (dt={dt_viz_over})...")
    num_steps_over = int(T_total_hist / dt_viz_over)
    integrator = OverdampedLangevin(potential, gamma=gamma, kT=kT, dt=dt_viz_over)
    trajs_o, _, shadow_works_o = integrator.run_protocol(x_init, torch.zeros(num_steps_over).to(device))
    
    final_x_o = trajs_o[-1]
    final_w_o = shadow_works_o[-1]
    
    # Reweighting (Overdamped)
    log_weights_o = -beta * final_w_o
    valid_o = torch.isfinite(log_weights_o) & ~torch.isnan(final_x_o)
    log_weights_o[~valid_o] = -1e10
    log_weights_norm_o = log_weights_o - torch.logsumexp(log_weights_o, dim=0)
    weights_o = torch.exp(log_weights_norm_o) * len(log_weights_o)
    ess_over = (weights_o.sum()**2) / (weights_o**2).sum()
    
    # 1b. Shadow Work Simulation (Underdamped BAOAB)
    print(f"Running Underdamped BAOAB Simulation (dt={dt_viz_baoab})...")
    num_steps_baoab = int(T_total_hist / dt_viz_baoab)
    v_init = (torch.randn_like(x_init) * np.sqrt(kT)).to(device)
    baoab = UnderdampedLangevin(potential, gamma=gamma, kT=kT, dt=dt_viz_baoab)
    trajs_b, shadow_works_b = baoab.run_protocol(x_init, v_init, torch.zeros(num_steps_baoab).to(device))
    
    final_x_b = trajs_b[-1]
    final_w_b = shadow_works_b[-1]
    
    # Reweighting (BAOAB)
    log_weights_b = -beta * final_w_b
    valid_b = torch.isfinite(log_weights_b) & ~torch.isnan(final_x_b)
    log_weights_b[~valid_b] = -1e10
    log_weights_n_b = log_weights_b - torch.logsumexp(log_weights_b, dim=0)
    weights_b = torch.exp(log_weights_n_b) * len(log_weights_b)
    ess_baoab = (weights_b.sum()**2) / (weights_b**2).sum()
    
    # Plot Histograms
    bins = np.linspace(-2.5, 2.5, 50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Move results to CPU for plotting
    x_plot_cpu = x_plot.cpu().numpy()
    p_true_cpu = p_true.cpu().numpy()
    final_x_o_cpu = final_x_o.cpu()
    weights_o_cpu = weights_o.cpu()
    valid_o_cpu = valid_o.cpu().numpy()
    final_x_b_cpu = final_x_b.cpu()
    weights_b_cpu = weights_b.cpu()
    valid_b_cpu = valid_b.cpu().numpy()

    # Subplot 1: Overdamped
    ax_hist_shadow.plot(x_plot_cpu, p_true_cpu, 'k--', linewidth=2, label='True Boltzmann', zorder=10)
    
    # Biased
    counts_biased_o, _ = np.histogram(final_x_o_cpu[valid_o_cpu].numpy(), bins=bins, density=True)
    ax_hist_shadow.plot(bin_centers, counts_biased_o, color=c_over, linewidth=1.5, linestyle=':', label=f'Overdamped Biased (dt={dt_viz_over})')
    
    # Debiased
    counts_corr_o, _ = np.histogram(final_x_o_cpu[valid_o_cpu].numpy(), bins=bins, weights=weights_o_cpu[valid_o_cpu].numpy(), density=True)
    ax_hist_shadow.plot(bin_centers, counts_corr_o, color=c_over, linewidth=2.5, linestyle='-', label=f'Overdamped Debiased (ESS={int(ess_over.item())})')
    
    ax_hist_shadow.set_title("Overdamped Langevin Debiasing", fontweight='bold')
    ax_hist_shadow.legend(loc='upper right', fontsize=8)
    ax_hist_shadow.set_xlabel("x")
    ax_hist_shadow.set_ylabel("Density")

    # Subplot 2: BAOAB
    ax_hist_mala.plot(x_plot_cpu, p_true_cpu, 'k--', linewidth=2, label='True Boltzmann', zorder=10)
    
    # Biased
    counts_biased_b, _ = np.histogram(final_x_b_cpu[valid_b_cpu].numpy(), bins=bins, density=True)
    ax_hist_mala.plot(bin_centers, counts_biased_b, color=c_baoab, linewidth=1.5, linestyle=':', label=f'BAOAB Biased (dt={dt_viz_baoab})')
    
    # Debiased
    counts_corr_b, _ = np.histogram(final_x_b_cpu[valid_b_cpu].numpy(), bins=bins, weights=weights_b_cpu[valid_b_cpu].numpy(), density=True)
    ax_hist_mala.plot(bin_centers, counts_corr_b, color=c_baoab, linewidth=2.5, linestyle='-', label=f'BAOAB Debiased (ESS={int(ess_baoab.item())})')
    
    ax_hist_mala.set_title("BAOAB Underdamped Debiasing", fontweight='bold')
    ax_hist_mala.legend(loc='upper right', fontsize=8)
    ax_hist_mala.set_xlabel("x")
    
    # --- PART 2: EFFICIENCY SWEEP ---
    print("\n--- Running Efficiency Sweep ---")
    dt_values = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.24, 0.32, 0.40, 0.50, 0.60, 0.80]
    
    eff_mala_list, eff_shadow_list, eff_rwm_list = [], [], []
    eff_baoab_list = [] 
    ar_mala_list, ar_rwm_list = [], []
    shadow_accumulation_rates_over = [] 
    shadow_accumulation_rates_baoab = []
    
    x_sweep = x_init[:2000].to(device)
    v_sweep = (torch.randn_like(x_sweep) * np.sqrt(kT)).to(device) # Init velocities
    
    for dt in dt_values:
        ns = int(T_total_eff / dt)
        if ns < 10: ns = 10
        
        # MALA
        mala = MALAIntegrator(potential, gamma=gamma, kT=kT, dt=dt)
        tm, am = mala.run(x_sweep, ns)
        eff_mala_list.append(calculate_ess_dynamic(tm.cpu()) / ns)
        ar_mala_list.append(am.mean().item())
        
        # Shadow (Overdamped)
        integ = OverdampedLangevin(potential, gamma=gamma, kT=kT, dt=dt)
        _, _, sw = integ.run_protocol(x_sweep, torch.zeros(ns).to(device))
        fw = sw[-1]
        shadow_accumulation_rates_over.append(fw.mean().item() / T_total_eff)
        
        lw = -beta * fw
        if torch.isnan(lw).any() or torch.isinf(lw).any():
            ess_s = 1e-10
        else:
            lw_n = lw - torch.logsumexp(lw, dim=0)
            ws = torch.exp(lw_n)
            ess_s = 1.0 / torch.sum(ws**2).item()
        eff_shadow_list.append(ess_s / ns)
        
        # RWM
        rwm = RWMIntegrator(potential, gamma=gamma, kT=kT, dt=dt)
        tr, ar = rwm.run(x_sweep, ns)
        eff_rwm_list.append(calculate_ess_dynamic(tr.cpu()) / ns)
        ar_rwm_list.append(ar.mean().item())
        
        # BAOAB (Underdamped)
        baoab = UnderdampedLangevin(potential, gamma=gamma, kT=kT, dt=dt)
        tb_traj, sw_b = baoab.run_protocol(x_sweep, v_sweep, torch.zeros(ns).to(device))
        eff_baoab_list.append(calculate_ess_dynamic(tb_traj.cpu()) / ns)
        shadow_accumulation_rates_baoab.append(sw_b[-1].mean().item() / T_total_eff)
        
        print(f"dt={dt:.3f} | Shd={eff_shadow_list[-1]:.1e} | MALA={eff_mala_list[-1]:.1e} | BAOAB={eff_baoab_list[-1]:.1e}")

    # Plot Efficiency
    ax_eff.plot(dt_values, eff_mala_list, 'o-', color=c_mala, label='MALA', linewidth=2)
    ax_eff.plot(dt_values, eff_shadow_list, 's-', color=c_over, label='Langevin (Shadow reweighting)', linewidth=2)
    ax_eff.plot(dt_values, eff_rwm_list, '^-', color=c_rwm, label='RWM', linewidth=2, linestyle='--')
    ax_eff.plot(dt_values, eff_baoab_list, 'D-', color=c_baoab, label='BAOAB (Shadow reweighting)', linewidth=2)
    
    ax_eff.set_xlabel(r"Time Step $\Delta t$", fontweight='bold')
    ax_eff.set_ylabel("Efficiency (ESS / Step)", fontweight='bold')
    ax_eff.set_title("Sampling Efficiency Comparison", fontweight='bold')
    ax_eff.set_yscale('log')
    ax_eff.set_ylim(bottom=1e-4)
    ax_eff.grid(True, alpha=0.3, which='both')
    ax_eff.legend()
    
    # Plot Accumulation & Acceptance
    ax_acc.plot(dt_values, ar_mala_list, 'o-', color=c_mala, label='MALA Acceptance')
    ax_acc.plot(dt_values, ar_rwm_list, '^-', color=c_rwm, label='RWM Acceptance', linestyle='--')
    
    ax_acc2 = ax_acc.twinx()
    ax_acc2.plot(dt_values, shadow_accumulation_rates_over, 's:', color=c_over, label='Overdamped Work Rate', linewidth=1.5)
    ax_acc2.plot(dt_values, shadow_accumulation_rates_baoab, 'D:', color=c_baoab, label='BAOAB Work Rate', linewidth=1.5)
    ax_acc2.set_ylabel(r"Shadow Work Rate ($\langle W \rangle / T$)", fontweight='bold')
    
    ax_acc.set_xlabel(r"Time Step $\Delta t$", fontweight='bold')
    ax_acc.set_ylabel("Acceptance Probability", fontweight='bold')
    ax_acc.set_title("Diagnostics: Acceptance & Work Rate", fontweight='bold')
    ax_acc.set_ylim(0, 1.1)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc='center left', fontsize=9)
    
    # --- PART 3: SHADOW WORK ACCUMULATION ---
    print("\n--- Generating Accumulation Plots ---")
    
    # Baseline comparison
    dt_baseline = 0.01
    
    for dt in [dt_baseline, dt_viz_over, dt_viz_baoab]:
        ns = int(T_total_hist / dt)
        t_axis = np.arange(ns) * dt
        alpha = 0.3 if dt == dt_baseline else 1.0
        lw = 1.5 if dt == dt_baseline else 2.5
        
        # Overdamped: only plot for small/medium steps to avoid blowup
        if dt <= 0.05:
            integ = OverdampedLangevin(potential, gamma=gamma, kT=kT, dt=dt)
            batch_accum = 1000
            x_start = x_init[:batch_accum].to(device)
            _, _, sw = integ.run_protocol(x_start, torch.zeros(ns).to(device))
            mean_w = sw.mean(dim=1).cpu().numpy()
            ax_accum.plot(t_axis, mean_w, color=c_over, linewidth=lw, alpha=alpha, label=f'Overdamped (dt={dt})')
        
        # Underdamped: plot for all (including the large step)
        baoab = UnderdampedLangevin(potential, gamma=gamma, kT=kT, dt=dt)
        v_start = (torch.randn(1000, device=device) * np.sqrt(kT)).to(device) # Small batch for accum
        x_start_b = x_init[:1000].to(device)
        _, sw_b = baoab.run_protocol(x_start_b, v_start, torch.zeros(ns).to(device))
        mean_w_b = sw_b.mean(dim=1).cpu().numpy()
        ax_accum.plot(t_axis, mean_w_b, color=c_baoab, linestyle='--', linewidth=lw, alpha=alpha, label=f'BAOAB (dt={dt})')
    ax_accum.set_xlabel("Time", fontweight='bold')
    ax_accum.set_ylabel(r"$\langle W_{shadow} \rangle$", fontweight='bold')
    ax_accum.set_title("Shadow Work Accumulation (Linear Regime)", fontweight='bold')
    ax_accum.legend()
    ax_accum.grid(True, alpha=0.3)
    
    # Save
    plot_path = os.path.join(assets_dir, "work_reweighting.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nDashboard saved to {plot_path}")

if __name__ == "__main__":
    run_dashboard_demo()
