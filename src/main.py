#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Importing libraries and modules
import numpy as np
import scipy.fft as sfft
import matplotlib.pyplot as plt

class advection_diffusion:

    def __init__(self, 
                 N: int = 64, 
                 L: float = 4 * np.pi, 
                 T: float = 1.0, 
                 dt: float = 1e-2,
                 v: float = 1.0,
                 D: float = 1.0
                 ) -> None:
        # Defining spacial domain
        self.x = np.linspace(0, L, N)
        # Defining time array
        self.t = np.arange(0, T, dt)
        self.dt = dt
        self.L = L

        # Defining wavenumbers
        self.k = np.fft.fftfreq(N, d = L / N) * 2 * np.pi
        
        # Computing linear operator in Fourier space (Advection + Diffusion terms)
        self.L_hat = -(1j * v * self.k + D * self.k**2)

        # Initialize u(x, t)
        self.u = np.zeros((self.x.shape[0], self.t.shape[0]))

    def u0(self, l: float = 0.1, amp: float = 1.0) -> np.ndarray:
        """
        Inintal conditions; Gaussian distribution
        """
        return amp * np.exp(-((self.x - (self.L / 2))**2 ) / (2 * l**2))

    def solve(self) -> np.ndarray:
        """
        Solving the advection-diffusion equation using Fourier-Galerkin in space
        and backward difference in time
        """
        # Initial condition u(x, 0)
        self.u[:, 0] = self.u0()
        # Fourier transfrom of initial condition
        uhat = sfft.fft(self.u[:, 0])

        for i in range(self.t.shape[0]):
            uhat = uhat / (1 - self.dt * self.L_hat)
            self.u[:, i] = np.real(sfft.ifft(uhat))
        
        return self.u

    def animate(self) -> None:
        """
        Animate the solution of the 1D advection - diffusion equation using dirichlet boundary
        """
        plt.figure()
        for i in range(self.t.shape[0]):
            plt.clf()
            plt.plot(self.x, self.u[:, i])
            plt.xlabel(r'$x$')
            plt.ylabel(r'$u(x, t)$')
            plt.pause(1E-7)
        plt.show()

if __name__ == '__main__':
    ad = advection_diffusion()
    ad.solve()
    ad.animate()