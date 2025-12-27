# Shadow Work Correction for Jarzynski Free Energy Estimation

Based on: Sivak, D. A., Chodera, J. D., & Crooks, G. E. (2013). *Physical Review X*.

## Overview

The core insight from Sivak & Crooks is that **numerical integrator error is not random noise—it's work done on a "shadow Hamiltonian"**. This means we can track and correct for discretization error in non-equilibrium free energy calculations.

---

## The Problem: Two Sources of Bias

In non-equilibrium free energy calculations (e.g., pulling a molecule), we face two limitations:

1. **Physical dissipation**: Fast switching → large dissipation → $\langle W \rangle > \Delta F$
2. **Discretization error**: Large Δt → integrator bias → wrong statistics

The Jarzynski equality relates work to free energy:

$$\Delta F = -\frac{1}{\beta} \ln \langle e^{-\beta W} \rangle$$

Shadow work correction addresses discretization by including integrator error in the work:

$$\Delta F = -\frac{1}{\beta} \ln \langle e^{-\beta (W_{protocol} + W_{shadow})} \rangle$$

---

## Results: Shadow Work Correction Dashboard

![Jarzynski Shadow Work Dashboard](../assets/jarzynski_shadow_work.png)

### Key Observations

**Left column** (Non-equilibrium Protocol at Δt = 0.05):
- Potential tilts from symmetric to asymmetric (λ: 0 → 1)
- Work distributions show protocol work (red) vs total work (teal)
- **Solid lines**: Jarzynski estimates (what we use for ΔF)
- **Dotted lines**: Arithmetic means (always ≥ ΔF by second law)

**Right column** (Time Step Sweep):
- **Naive estimate** (protocol work only) drifts from true ΔF as Δt increases
- **Corrected estimate** (+ shadow work) stays accurate for moderate Δt
- ESS decreases with larger Δt due to weight variance
- Sweet spot: Δt ~ 0.03-0.04 gives good accuracy with reasonable ESS

### Why Jarzynski ≠ Mean

The Jarzynski estimate uses **exponential averaging**, not arithmetic mean:
- Arithmetic mean: $\langle W \rangle$ — always overestimates ΔF
- Jarzynski estimate: $-k_BT \ln \langle e^{-\beta W} \rangle$ — gives exact ΔF

This is why the solid lines (Jarzynski) differ from dotted lines (means) in the work distribution plot.

---

## Animation: Non-equilibrium Protocol

![Jarzynski Protocol Animation](../assets/jarzynski_protocol.gif)

The animation shows:
1. Potential tilting from symmetric (λ=0) to asymmetric (λ=1)
2. Particle distribution following the changing landscape
3. Work distributions evolving (protocol vs total)
4. Jarzynski estimate converging to true ΔF

---

## Trade-offs

| Time Step | Naive Error | Corrected Error | ESS | Shadow Work |
|-----------|-------------|-----------------|-----|-------------|
| 0.01 | Small | ~0 | High | Low |
| 0.03 | Moderate | ~0 | Good | Moderate |
| 0.05 | Large | Small | OK | High |
| 0.06+ | Very Large | Variable | Low | Very High |

**Conclusion**: Shadow work correction extends the usable range of time steps, but there's still a practical limit where weight variance becomes too large.

---

## Code

```bash
python scripts/demo_shadow_jarzynski.py
```

Generates:
- `assets/jarzynski_shadow_work.png` — Combined dashboard
- `assets/jarzynski_protocol.gif` — Animation

