import torch
import torch.nn as nn

class HarmonicPotential(nn.Module):
    """
    Harmonic potential U(x, lambda) = 0.5 * k * (x - x_center(lambda))^2
    where x_center(lambda) = (1 - lambda) * x_start + lambda * x_end
    """
    def __init__(self, k=10.0, x_start=0.0, x_end=1.0):
        super().__init__()
        self.k = k
        self.x_start = x_start
        self.x_end = x_end

    def get_center(self, lmbda):
        """Compute the trap center for a given lambda."""
        return (1.0 - lmbda) * self.x_start + lmbda * self.x_end

    def forward(self, x, lmbda):
        """Compute the potential energy."""
        center = self.get_center(lmbda)
        return 0.5 * self.k * (x - center)**2

    def force(self, x, lmbda):
        """Compute the force -dU/dx."""
        # For harmonic potential: -k * (x - center)
        center = self.get_center(lmbda)
        return -self.k * (x - center)

    def dU_dlambda(self, x, lmbda):
        """Compute the partial derivative of potential w.r.t. lambda."""
        # U = 0.5 * k * (x - ((1-lmbda)*x_s + lmbda*x_e))^2
        # dU/dlambda = k * (x - center) * (-d_center/dlambda)
        # d_center/dlambda = -x_start + x_end
        center = self.get_center(lmbda)
        d_center_dlambda = self.x_end - self.x_start
        return -self.k * (x - center) * d_center_dlambda

class DoubleWellPotential(nn.Module):
    """
    Double well potential U(x) = a*x^4 - b*x^2 + c*x
    where c = c(lambda) = (1 - lambda) * c_start + lambda * c_end
    """
    def __init__(self, a=1.0, b=2.0, c_start=0.0, c_end=0.0):
        super().__init__()
        self.a = a
        self.b = b
        self.c_start = c_start
        self.c_end = c_end

    def get_c(self, lmbda):
        return (1.0 - lmbda) * self.c_start + lmbda * self.c_end

    def forward(self, x, lmbda):
        c = self.get_c(lmbda)
        return self.a * x**4 - self.b * x**2 + c * x

    def force(self, x, lmbda):
        c = self.get_c(lmbda)
        return -(4.0 * self.a * x**3 - 2.0 * self.b * x + c)

    def dU_dlambda(self, x, lmbda):
        # U = a*x^4 - b*x^2 + ((1-lmbda)*c_s + lmbda*c_e)*x
        # dU/dlambda = (c_end - c_start) * x
        return (self.c_end - self.c_start) * x
