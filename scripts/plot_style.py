"""
Shared plotting style for noneq-work demos.
Consistent monospace aesthetic across all scripts.
"""
import matplotlib.pyplot as plt

# Color palette (consistent across demos)
COLORS = {
    'primary': '#1f77b4',      # Blue - main/cold/accurate
    'secondary': '#ff7f0e',    # Orange - overdamped/warm
    'tertiary': '#9467bd',     # Purple - BAOAB/underdamped
    'accent': '#2ca02c',       # Green - estimates/corrected
    'warning': '#d62728',      # Red - naive/error
    'neutral': '#7f7f7f',      # Gray
    'dark': '#2E4057',         # Dark blue-gray
    'teal': '#048A81',         # Teal - corrected/debiased
}

# Semantic aliases for Jarzynski-specific plots
COLORS_JARZYNSKI = {
    'naive': '#d62728',        # Red for naive/uncorrected
    'corrected': '#048A81',    # Teal for shadow-corrected
    'true': 'black',           # Black for true values
    'protocol': '#ff7f0e',     # Orange for protocol work
    'shadow': '#2E4057',       # Dark for shadow work
    'ess': '#F19A3E',          # Golden for ESS
}

def apply_style():
    """Apply consistent plotting style across all demos."""
    plt.rcParams.update({
        # Font settings - monospace for technical/research feel
        "font.family": "monospace",
        "font.monospace": ["DejaVu Sans Mono", "Menlo", "Consolas", "Monaco"],
        "font.size": 11,
        
        # Axes
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 8.0,
        "axes.labelpad": 4.0,
        
        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        
        # Grid
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        
        # Legend
        "legend.fontsize": 9,
        "legend.frameon": False,
        
        # Figure
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 150,
        
        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


def style_jarzynski_work_plot(ax, dF_true, dF_naive, dF_corrected, 
                               mean_prot=None, mean_total=None,
                               show_means=True, show_jarzynski=True):
    """
    Add styled vertical lines for work distribution plots.
    
    - Solid colored lines: Jarzynski estimates (what we care about)
    - Dotted colored lines: arithmetic means (for reference)
    - Dashed black line: true ΔF
    """
    # True ΔF - always show
    ax.axvline(dF_true, color='black', linestyle='--', linewidth=2.5, 
               label=f'True ΔF = {dF_true:.2f}', zorder=10)
    
    # Jarzynski estimates (the important ones!)
    if show_jarzynski:
        ax.axvline(dF_naive, color=COLORS_JARZYNSKI['naive'], 
                   linestyle='-', linewidth=2.5, 
                   label=f'Jarzynski (naive) = {dF_naive:.2f}', zorder=9)
        ax.axvline(dF_corrected, color=COLORS_JARZYNSKI['corrected'], 
                   linestyle='-', linewidth=2.5, 
                   label=f'Jarzynski (corrected) = {dF_corrected:.2f}', zorder=9)
    
    # Arithmetic means (secondary - for comparison with Jarzynski)
    if show_means and mean_prot is not None:
        ax.axvline(mean_prot, color=COLORS_JARZYNSKI['naive'], 
                   linestyle=':', linewidth=1.5, alpha=0.6,
                   label=f'⟨W_prot⟩ = {mean_prot:.2f}')
    if show_means and mean_total is not None:
        ax.axvline(mean_total, color=COLORS_JARZYNSKI['corrected'], 
                   linestyle=':', linewidth=1.5, alpha=0.6,
                   label=f'⟨W_total⟩ = {mean_total:.2f}')

