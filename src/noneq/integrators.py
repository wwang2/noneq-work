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
        Protocol:
        1. Change lambda: W_protocol = U(x, lmbda + dlmbda) - U(x, lmbda)
        2. Langevin update: Shadow work calculation
        """
        # 1. Update lambda and track protocol work
        u_old = self.potential(x, lmbda)
        new_lmbda = lmbda + dlmbda
        u_new_lmbda = self.potential(x, new_lmbda)
        
        # dU/dlambda * dlmbda is a first-order approximation
        # Using exact difference is more robust for protocol work
        work_protocol = u_new_lmbda - u_old
        
        # 2. Langevin update (thermal relaxation)
        f_old = self.potential.force(x, new_lmbda)
        noise = torch.randn_like(x) * np.sqrt(2 * self.kT * self.dt / self.gamma)
        new_x = x + (f_old / self.gamma) * self.dt + noise
        
        # 3. Shadow work calculation for the Langevin step
        # beta * w_shadow = ln(pi(x)/pi(x')) + ln(P(x'|x)/P(x|x'))
        f_new = self.potential.force(new_x, new_lmbda)
        u_final = self.potential(new_x, new_lmbda)
        
        # Log-ratio of transition probabilities: ln(P(x'|x)/P(x|x'))
        # Using the formula derived from Gaussian transition kernels:
        # ln(P'/P) = 1/(2*kT) * (dx) * (f_old + f_new) - dt/(4*kT*gamma) * (f_old^2 - f_new^2)
        dx = new_x - x
        log_prob_ratio = (1.0 / (2.0 * self.kT)) * dx * (f_old + f_new) - \
                         (self.dt / (4.0 * self.kT * self.gamma)) * (f_old**2 - f_new**2)
        
        # beta * w_shadow = beta * (U(x') - U(x)) + log_prob_ratio
        # Here U(x) is u_new_lmbda (after lambda update but before x update)
        beta_w_shadow = self.beta * (u_final - u_new_lmbda) + log_prob_ratio
        work_shadow = beta_w_shadow * self.kT
        
        return new_x, new_lmbda, work_protocol, work_shadow

    def run_protocol(self, x_init, lambdas):
        """
        Run a full protocol given a sequence of lambdas.
        lambdas: tensor of shape (num_steps,)
        x_init: batch of initial positions (num_trajectories,)
        """
        num_steps = len(lambdas)
        x = x_init.clone()
        total_protocol_work = torch.zeros_like(x)
        total_shadow_work = torch.zeros_like(x)
        
        # Keep track of trajectories for visualization
        trajectories = [x.clone()]
        protocol_works = [total_protocol_work.clone()]
        shadow_works = [total_shadow_work.clone()]
        
        for i in range(num_steps - 1):
            lmbda = lambdas[i]
            dlmbda = lambdas[i+1] - lmbda
            x, _, dw_p, dw_s = self.step(x, lmbda, dlmbda)
            total_protocol_work += dw_p
            total_shadow_work += dw_s
            trajectories.append(x.clone())
            protocol_works.append(total_protocol_work.clone())
            shadow_works.append(total_shadow_work.clone())
            
        return torch.stack(trajectories), torch.stack(protocol_works), torch.stack(shadow_works)

class UnderdampedLangevin:
    """
    Underdamped Langevin using BAOAB splitting.
    """
    def __init__(self, potential, gamma=1.0, kT=1.0, dt=0.01, mass=1.0):
        self.potential = potential
        self.gamma = gamma
        self.kT = kT
        self.dt = dt
        self.mass = mass
        
        # Constants for O-step (Ornstein-Uhlenbeck)
        self.alpha = np.exp(-gamma * dt)
        self.sigma_v = np.sqrt(kT / mass * (1 - self.alpha**2))
        
    def step(self, x, v, lmbda):
        # B: Velocity half-step
        force = self.potential.force(x, lmbda)
        v = v + (self.dt / 2) * force / self.mass
        
        # A: Position half-step
        x = x + (self.dt / 2) * v
        
        # O: Noise step on velocity
        noise = torch.randn_like(v) * self.sigma_v
        v = self.alpha * v + noise
        
        # A: Position half-step
        x = x + (self.dt / 2) * v
        
        # B: Velocity half-step
        force = self.potential.force(x, lmbda) # Force at new x
        v = v + (self.dt / 2) * force / self.mass
        
        return x, v
        
    def run_protocol(self, x_init, v_init, lambdas):
        num_steps = len(lambdas)
        x = x_init.clone()
        v = v_init.clone()
        trajectories = [x.clone()]
        
        for i in range(num_steps - 1):
            lmbda = lambdas[i]
            x, v = self.step(x, v, lmbda)
            trajectories.append(x.clone())
            
        return torch.stack(trajectories)
