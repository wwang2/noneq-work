import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm
from noneq.potentials import HarmonicPotential
from noneq.integrators import OverdampedLangevin
from noneq.estimators import jarzynski_estimate, jarzynski_convergence

# Applying plotting style from rules
plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["DejaVu Sans Mono", "Menlo", "Consolas", "Monaco"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 8.0,
    "axes.labelpad": 4.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

def run_demo():
    # 1. Setup Parameters
    k = 10.0
    x_start, x_end = 0.0, 2.0
    kT = 1.0
    beta = 1.0 / kT
    gamma = 1.0
    dt = 0.01
    num_steps = 200
    num_trajectories = 2000 
    
    potential = HarmonicPotential(k=k, x_start=x_start, x_end=x_end)
    integrator = OverdampedLangevin(potential, gamma=gamma, kT=kT, dt=dt)
    
    # Protocols
    lambdas_fwd = torch.linspace(0, 1, num_steps)
    lambdas_rev = torch.linspace(1, 0, num_steps)
    
    # 2. Run Forward Protocol
    print(f"Running Forward Protocol with {num_trajectories} samples...")
    x_init_fwd = torch.randn(num_trajectories) * np.sqrt(kT / k) + x_start
    traj_fwd, works_fwd, _ = integrator.run_protocol(x_init_fwd, lambdas_fwd)
    final_works_fwd = works_fwd[-1]
    
    # 3. Run Reverse Protocol
    print(f"Running Reverse Protocol with {num_trajectories} samples...")
    x_init_rev = torch.randn(num_trajectories) * np.sqrt(kT / k) + x_end
    traj_rev, works_rev, _ = integrator.run_protocol(x_init_rev, lambdas_rev)
    final_works_rev = works_rev[-1]
    
    true_delta_f = 0.0
    
    # Pre-calculate convergence with uncertainty for the animation
    print("Computing convergence and uncertainty...")
    conv_indices, conv_fwd, conv_errors = jarzynski_convergence(final_works_fwd, beta, compute_error=True)
    
    # 4. Generate Combined GIF
    print("Generating Animation...")
    frames = []
    assets_dir = "/Users/wujiewang/projects/noneq-work/assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    # Select a subset of trajectories to plot for clarity
    plot_indices = np.random.choice(num_trajectories, 20, replace=False)
    
    # Time axis
    times = np.arange(num_steps) * dt
    
    # Pre-calculate work bounds for consistent coloring
    all_works = torch.cat([works_fwd, -works_rev])
    w_min, w_max = all_works.min().item(), all_works.max().item()

    for i in tqdm(range(0, num_steps, 5)):
        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        
        # --- Top: Trajectories ---
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        current_w = works_fwd[i]
        
        for idx in plot_indices:
            # Color changes as a function of work at current time t
            w_val = current_w[idx].item()
            w_norm = (w_val - w_min) / (w_max - w_min)
            color = plt.cm.viridis(w_norm)
            ax1.plot(times[:i+1], traj_fwd[:i+1, idx], color=color, alpha=0.7, linewidth=1.5)
        
        # Plot trap center
        ax1.plot(times[:i+1], potential.get_center(lambdas_fwd[:i+1]), 'k--', alpha=0.8, label='Trap Center')
        ax1.set_title("Forward Trajectories (colored by current work)", fontweight='bold')
        ax1.set_xlabel("Time", fontweight='bold')
        ax1.set_ylabel("Position", fontweight='bold')
        ax1.set_xlim(0, times[-1])
        ax1.set_ylim(x_start - 1.5, x_end + 1.5)
        ax1.legend()
        ax1.set_axisbelow(True)

        # --- Bottom Left: Work Distributions ---
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        current_w_fwd = works_fwd[i]
        current_w_rev = -works_rev[i] 
        
        bins = np.linspace(w_min - 0.5, w_max + 0.5, 40)
        
        ax2.hist(current_w_fwd.numpy(), bins=bins, density=True, alpha=0.5, label=r'$P_F(W)$', color='#1f77b4')
        ax2.hist(current_w_rev.numpy(), bins=bins, density=True, alpha=0.5, label=r'$-P_R(-W)$', color='#d62728')
        ax2.axvline(true_delta_f, color='black', linewidth=2, label=r'True $\Delta F$')
        
        if i > 0:
            df_est = (jarzynski_estimate(current_w_fwd, beta) - jarzynski_estimate(-current_w_rev, beta)) / 2.0
            ax2.axvline(df_est.item(), color='green', linestyle='--', linewidth=2, label=r'Est $\Delta F$')
            
        ax2.set_title(f"Work Distributions (t={times[i]:.2f})", fontweight='bold')
        ax2.set_xlabel("Work", fontweight='bold')
        ax2.set_ylabel("Density", fontweight='bold')
        ax2.legend()
        ax2.set_axisbelow(True)

        # --- Bottom Right: Convergence ---
        ax3 = plt.subplot2grid((2, 2), (1, 1))
        # N samples scales with simulation progress for visual effect
        n_samples_limit = int((i + 1) / num_steps * num_trajectories)
        
        # Filter convergence data for the animation frame
        mask = conv_indices <= n_samples_limit
        if np.any(mask):
            idx_sub = conv_indices[mask]
            est_sub = conv_fwd[mask]
            err_sub = conv_errors[mask]
            
            ax3.plot(idx_sub, est_sub, color='#1f77b4', linewidth=2, label='Jarzynski Estimate')
            # Add uncertainty band
            ax3.fill_between(idx_sub, est_sub - err_sub, est_sub + err_sub, color='#1f77b4', alpha=0.2, label=r'1$\sigma$ Uncertainty')
            
            ax3.axhline(true_delta_f, color='black', linewidth=1.5, label=r'True $\Delta F$')
            ax3.set_xlim(0, num_trajectories)
            ax3.set_ylim(true_delta_f - 1.5, true_delta_f + 1.5)
            ax3.set_title(f"Convergence (N={n_samples_limit})", fontweight='bold')
            ax3.set_xlabel("Number of Samples", fontweight='bold')
            ax3.set_ylabel(r"$\Delta F$ Estimate", fontweight='bold')
            ax3.legend()
            ax3.set_axisbelow(True)

        # Convert plot to image
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())
        image = image[:, :, :3]
        frames.append(image)
        plt.close()

    gif_path = os.path.join(assets_dir, "fluctuation_demo.gif")
    imageio.v2.mimsave(gif_path, frames, fps=10)
    print(f"Animation saved to {gif_path}")

if __name__ == "__main__":
    run_demo()
