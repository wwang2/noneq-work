import torch
import numpy as np

class OverdampedLangevin:
    """
    Overdamped Langevin integrator:
    dx = (force / gamma) * dt + sqrt(2 * kT * dt / gamma) * dW
    """
    def __init__(self, potential, gamma=1.0, kT=1.0, dt=0.01):
        self.potential = potential
        self.gamma = gamma
        self.kT = kT
        self.dt = dt
        self.beta = 1.0 / kT

    def step(self, x, lmbda, dlmbda=0.0):
        """
        Perform one integration step and track work.
        Work is defined as dW = (dU/dlambda) * dlambda
        """
        # 1. Update lambda and track work
        # Work is done by changing the external parameter lambda
        # Protocol: x stays fixed during dlambda change (standard convention)
        work_step = self.potential.dU_dlambda(x, lmbda) * dlmbda
        
        # 2. Update lambda
        new_lmbda = lmbda + dlmbda
        
        # 3. Langevin update (thermal relaxation)
        force = self.potential.force(x, new_lmbda)
        noise = torch.randn_like(x) * np.sqrt(2 * self.kT * self.dt / self.gamma)
        new_x = x + (force / self.gamma) * self.dt + noise
        
        return new_x, new_lmbda, work_step

    def run_protocol(self, x_init, lambdas):
        """
        Run a full protocol given a sequence of lambdas.
        lambdas: tensor of shape (num_steps,)
        x_init: batch of initial positions (num_trajectories,)
        """
        num_steps = len(lambdas)
        x = x_init.clone()
        total_work = torch.zeros_like(x)
        
        # Keep track of trajectories for visualization
        trajectories = [x.clone()]
        works = [total_work.clone()]
        
        for i in range(num_steps - 1):
            lmbda = lambdas[i]
            dlmbda = lambdas[i+1] - lmbda
            x, _, dw = self.step(x, lmbda, dlmbda)
            total_work += dw
            trajectories.append(x.clone())
            works.append(total_work.clone())
            
        return torch.stack(trajectories), torch.stack(works), total_work
