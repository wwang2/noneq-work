# Shadow Work Correction

Notes on: Sivak, D. A., Chodera, J. D., & Crooks, G. E. (2013). Using nonequilibrium fluctuation theorems to understand and correct errors in equilibrium and nonequilibrium simulations of discrete Langevin dynamics. Physical Review X.

[Link to Paper](https://arxiv.org/pdf/1107.2967)

## Key Concepts

### 1. Finite Time Step as a Driven Process
Even when the physical Hamiltonian $H(x)$ is time-independent, a discrete-time integrator (like Langevin) does not sample from the true equilibrium distribution $\pi(x) \propto e^{-\beta H(x)}$. Instead, it can be viewed as a driven, nonequilibrium process.

### 2. Shadow Work Definitions

The "shadow work" $w_{shadow}$ accounts for the irreversibility introduced by discretization. For any transition $x \to x'$, it is generally defined by the log-ratio of forward and reverse transition probabilities relative to the stationary distribution:

$$\beta w_{shadow} = \ln \frac{\pi(x)}{\pi(x')} + \ln \frac{P(x'|x)}{P(x|x')}$$

#### A. Overdamped Langevin
For a single-step update $x_{n+1} = x_n + \frac{f(x_n)}{\gamma} \Delta t + \sqrt{\frac{2 k_B T \Delta t}{\gamma}} \eta$, the shadow work is calculated using the transition kernel of the Euler-Maruyama discretization:

$$\ln \frac{P(x'|x)}{P(x|x')} = \frac{1}{2 k_B T} (x' - x)(f(x) + f(x')) - \frac{\Delta t}{4 k_B T \gamma} (f(x)^2 - f(x')^2)$$

Substituting this into the general definition (where $\ln \frac{\pi(x)}{\pi(x')} = \beta [U(x') - U(x)]$), we get:
$$\beta w_{shadow} = \beta [U(x') - U(x)] + \ln \frac{P(x'|x)}{P(x|x')}$$

#### B. Underdamped Langevin (BAOAB)
In the BAOAB splitting scheme ($V \xrightarrow{B} X \xrightarrow{A} V \xrightarrow{O} X \xrightarrow{A} V \xrightarrow{B}$), the position updates ($A$) and deterministic velocity updates ($B$) are symplectic and reversible. Irreversibility is introduced solely in the **O-step** (the Ornstein-Uhlenbeck noise step).

The total shadow work for a BAOAB step is:
$$\beta w_{shadow} = \beta [H(x', v') - H(x, v)] + \ln \frac{P(v_{after} | v_{before})}{P(v_{before} | v_{after})}$$

where:
*   $H(x, v) = U(x) + \frac{1}{2} m v^2$ is the physical Hamiltonian.
*   $v_{before}, v_{after}$ are the velocities immediately before and after the stochastic **O-step**.
*   The log-prob ratio for the O-step velocity update ($v' = a v + \sigma \eta$) is:
    $$\ln \frac{P(v'|v)}{P(v|v')} = \frac{1}{2 \sigma^2} [ (v - a v')^2 - (v' - a v)^2 ]$$

### 3. Debiasing with Importance Weights
Expectation values under the true equilibrium distribution can be recovered from the biased simulation samples by weighting each sample $i$ by:

$$w_i = e^{-\beta W_{shadow, i}}$$

where $W_{shadow, i}$ is the accumulated shadow work along the trajectory.

## Demonstration Results

We simulated a **Double Well Potential** $U(x) = x^4 - 2x^2$ using an overdamped Langevin integrator with various time steps. This potential is stiff ($U'' \propto x^2$), making it sensitive to large time steps.

### Shadow Work Accumulation
As predicted in the paper (Fig 4), shadow work accumulates linearly over time in steady state. Larger time steps lead to faster accumulation, indicating greater discretization error and stronger "driving" by the integrator.

(See the bottom panel of the dashboard below for accumulation results).

### Equilibrium Distribution Correction
The uncorrected (biased) distributions for both Overdamped Langevin ($\Delta t=0.05$) and BAOAB ($\Delta t=0.5$) deviate from the true Boltzmann distribution. Applying the importance weights $e^{-\beta W_{shadow}}$ successfully recovers the true distribution, even when the effective sample size drops significantly due to variance accumulation. 

*Note: For the extreme step $\Delta t=0.5$, force clipping was implemented in the demonstration script to maintain numerical stability for the Double-Well potential, as standard integrators would otherwise diverge. We use **80,000 trajectories** and GPU acceleration (CUDA/Apple Silicon) to ensure smooth histograms even after weight collapse.*

## Comparison with MALA

The **Metropolis-Adjusted Langevin Algorithm (MALA)** uses the exact same "shadow work" quantity (log Metropolis ratio) but uses it to **reject** bad steps immediately, whereas our method **accumulates** it for later reweighting.

### Efficiency Comparison (Shadow Work vs. MALA vs. RWM vs. BAOAB)

We compared the Efficiency (ESS per simulation step) of four methods for a fixed physical time $T=50$.
-   **Langevin (Shadow reweighting)**: Overdamped Langevin.
-   **BAOAB (Shadow reweighting)**: Underdamped Langevin.
-   **MALA**: Metropolis-Adjusted Langevin Algorithm (Overdamped).
-   **RWM**: Random Walk Metropolis.

![Work Reweighting Dashboard](../assets/work_reweighting.png)

The dashboard includes:
1.  **Distributions**: Demonstrating successful debiasing for both Overdamped ($\Delta t=0.05$) and BAOAB ($\Delta t=0.5$) integrators. Biased samples (dotted lines) are corrected to match the Boltzmann distribution (solid lines). High sample counts ($N=80,000$) provide high-resolution results.
2.  **Efficiency Sweep**: Identifying the crossover point ($\Delta t \approx 0.02$) where MCMC methods become superior.
3.  **Accumulation**: Showing the linear growth of Shadow Work over time. The BAOAB integrator at $\Delta t=0.5$ shows how extreme error accumulates at large step sizes.

#### Key Findings
1.  **Small Steps ($\Delta t < 0.02$)**:
    *   **Shadow Work Wins**: It is ~20x more efficient than MALA. It allows the system to diffuse naturally without the slowdown caused by rejection or inertia.
2.  **Crossover ($\Delta t \approx 0.02$)**:
    *   Shadow work variance penalty equals the MCMC correlation penalty.
3.  **Large Steps ($\Delta t > 0.04$)**:
    *   **MCMC Wins**: Shadow work reweighting fails (ESS $\to 0$).
    *   **MALA** is remarkably robust, continuing to improve efficiency up to $\Delta t = 0.5$.
    *   **BAOAB** (Underdamped) is efficient but hits a stability limit at $\Delta t \approx 0.3$, causing NaNs.

**Conclusion**: Shadow Work reweighting is a specialized tool for **high-precision, small-timestep simulations** where it can extract exact equilibrium properties much faster than standard MCMC. For general coarse-grained exploration, MALA remains the most robust choice for this system.
