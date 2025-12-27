# Nonequilibrium Work Simulations & Notes

This repository contains simulations and notes exploring non-equilibrium thermodynamics, work relations, and shadow work.

## Structure

- `src/noneq/`: Core package for energy functions, integrators, and estimators.
- `notes/`: Theoretical background and paper summaries.
- `scripts/`: Simulation scripts and visualization notebooks.

## Demos

### Jarzynski & Crooks Fluctuation Theorems
![Fluctuation Theorems Demo](assets/fluctuation_demo.gif)

### Shadow Work & Debiasing
Demonstrating how finite-timestep discretization bias can be corrected using shadow work importance weights (based on [Sivak et al., 2013](https://arxiv.org/pdf/1107.2967)).

![Work Reweighting Dashboard](assets/work_reweighting.png)


## Key References

- [Sivak et al., 2011 (Shadow Work)](https://arxiv.org/pdf/1107.2967)
- [Jarzynski, 1997](https://arxiv.org/pdf/cond-mat/9610209)
- [Crooks, 1999](https://arxiv.org/pdf/cond-mat/9901352)
- [Sivak & Crooks, 2012 (Thermodynamic metrics)](https://arxiv.org/pdf/1201.4345)

## Installation

```bash
pip install -e .
```

## Running the Demo

To run the simulation and regenerate the animation:

```bash
python scripts/demo_fluctuation_theorems.py
```

