#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Importing libraries and modules
import numpy as np
import scipy.fft as sfft
import matplotlib.pyplot as plt

class advection_diffusion:

    def __init__(self, 
                 Nx: int = 128, 
                 Ny: int = 128,
                 Lx: float = 1.0, 
                 Ly: float = 1.0,
                 T: float = 1.0, 
                 dt: float = 1e-2,
                 vx: float = 0.5,
                 vy: float = 0.1,
                 D: float = 0.05,
                 dims: str = '1D'
                 ) -> None:
        # Storing dimensions
        self.dims = dims
        
        if dims == '1D':
            # Defining spacial domain
            self.x = np.linspace(0, Lx, Nx)
            # Defining time step
            self.dt = dt
            self.t = np.arange(0, T, dt)

            # Domain size
            self.L = Lx

            # Defining wavenumbers
            self.k = np.fft.fftfreq(Nx, d = Lx / Nx) * 2 * np.pi
            
            # Computing linear operator in Fourier space (Advection + Diffusion terms)
            self.L_hat = -(1j * vx * self.k + D * self.k**2)

            # Initialize u(x, t)
            self.u = np.zeros((self.x.shape[0], len(np.arange(0, T, dt))))
        elif dims == '2D':
            # Define spatial domain
            self.x = np.linspace(0, Lx, Nx)
            self.y = np.linspace(0, Ly, Ny)
            # Defining time array
            self.dt = dt
            self.t = np.arange(0, T, dt)

            # Defining wavenumbers (kx and ky)
            kx = np.fft.fftfreq(Nx, d = Lx / Nx) * 2 * np.pi
            ky = np.fft.fftfreq(Ny, d = Ly / Ny) * 2 * np.pi
            self.kx, self.ky = np.meshgrid(kx, ky)

            # Defining size of domain
            self.Lx = Lx
            self.Ly = Ly

            # Computing 2D linear operator in Fourier space (Advection + Diffusion terms) 
            self.L_hat = -((1j * vx * self.kx) + (1j * vy * self.ky) + D * (self.kx**2 + self.ky**2))

            # Initializing u(x, y, t)
            self.u = np.zeros((Nx, Ny, len(self.t)))


    def u0(self, l: float, amp: float = 1.0) -> np.ndarray:
        """
        Inintal conditions; Gaussian distribution
        """
        if self.dims == '1D':
            return amp * np.exp(-((self.x - (self.L / 2))**2 ) / (2 * l**2))
        elif self.dims == '2D':
            X, Y = np.meshgrid(self.x, self.y)
            return amp * np.exp(-( (X - (self.Lx / 2))**2 + (Y - (self.Ly / 2))**2 ) / (2 * l**2) )


    def solve(self) -> np.ndarray:
        """
        Solving the advection-diffusion equation using Fourier-Galerkin in space
        and backward Euler in time
        """
        if self.dims == '1D':
            # Initial condition u(x, 0)
            self.u[:, 0] = self.u0(0.03 * self.Lx) 
            # Fourier transfrom of initial condition
            uhat = sfft.fft(self.u[:, 0])

            # Looping over time t
            for i in range(1, self.u.shape[1]):
                uhat = uhat / (1 - self.dt * self.L_hat)
                self.u[:, i] = np.real(sfft.ifft(uhat))
        elif self.dims == '2D':
            # Initial condition u(x, y, 0)
            self.u[:, :, 0] = self.u0(0.03 * self.Lx)
            # Fourier transfrom of initial condition
            uhat = sfft.fft2(self.u[:, :, 0])

            # Looping over time t
            for i in range(1, self.u.shape[2]):
                uhat = uhat / (1 - self.dt * self.L_hat)
                self.u[:, :, i] = np.real(sfft.ifft2(uhat))
            
        return self.u

    def animate(self) -> None:
        """
        Animate the solution of the advection - diffusion equation using periodic boundary conditions in spatial domain
        """
        if self.dims == '1D':
            plt.figure()
            vmin, vmax = self.u.min(), self.u.max()
            for i in range(self.u.shape[1]):
                plt.clf()
                plt.plot(self.x, self.u[:, i])
                plt.xlabel(r'$x$')
                plt.ylabel(r'$u(x, t)$')
                plt.ylim([vmin, vmax])
                plt.pause(1E-7)
        elif self.dims == '2D':
            plt.figure()
            vmin, vmax = self.u.min(), self.u.max()
            for i in range(self.u.shape[2]):
                plt.clf()
                plt.contourf(self.x, self.y, self.u[:, :, i], levels = 200, vmin = vmin, vmax = vmax)
                plt.colorbar(label = r'$n(x, y)$')
                plt.xlabel(r'$x$')
                plt.ylabel(r'$y$')
                plt.title(f'Time: {self.t[i]:.3f}')
                plt.pause(1E-8)
        plt.show()

if __name__ == '__main__':
    ad = advection_diffusion(dims = '2D')
    ad.solve()
    ad.animate()