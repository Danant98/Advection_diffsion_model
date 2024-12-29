# Importing libraries and modules
using Plots, FFTW

struct AdvectionDiffusion
    Nx::Int
    Ny::Int
    Lx::Float64
    Ly::Float64
    T::Float64
    dt::Float64
    vx::Float64
    vy::Float64
    D::Float64
    x::Vector{Float64}
    y::Vector{Float64}
    kx::Matrix{Float64}
    ky::Matrix{Float64}
    Lhat::Matrix{ComplexF64}
    u::Array{Float64, 3}
    t::Vector{Float64}
end

# Constructor
function AdvectionDiffusion(Nx = 256, Ny = 256, Lx = 10.0, Ly = 10.0, T = 1.0, dt = 0.01, vx = 1.0, vy = 0.0, D = 1.0)
    # Creating spatial and temporal domains
    x = range(0, Lx, Nx)
    y = range(0, Ly, Ny)
    t = 0:dt:T

    # Wavenumbers
    kx = 2π * fftfreq(Nx, 1 / Nx)[1:(div(Nx, 2) + 1)] 
    ky = 2π * fftfreq(Ny, 1 / Ny)
    kx_grid = reshape(repeat(kx, inner = Ny), Nx, Ny)
    ky_grid = reshape(repeat(ky, outer = Nx), Nx, Ny)

    # Linear operator in Fourier space
    Lhat = -1im * (vx * kx_grid .+ vy * ky_grid) .- D * (kx_grid.^2 + ky_grid.^2)

    # Initializing u(x, y, t)
    u = zeros(Float64, Nx, Ny, length(t))

    AdvectionDiffusion(Nx, Ny, Lx, Ly, T, dt, vx, vy, D, x, y, kx_grid, ky_grid, Lhat, u, t)
end

# Inintal condition; Gaussian distribution
function Inintal_condition(model::AdvectionDiffusion, l::Float64, amp::Float64 = 1.0)
    # Creating 2D grid
    X = reshape(repeat(model.x, inner = model.Ny), model.Nx, model.Ny)
    Y = reshape(repeat(model.y, outer = model.Nx), model.Nx, model.Ny)
    amp * exp.(-((X .- model.Lx / 2).^2 .+ ( Y .- model.Ly / 2 ).^2 ) / (2 * l^2) )
end

# Solving using Fourier-Galerkin spectral method
function solve!(model::AdvectionDiffusion)
    # Initial. condtion
    model.u[:, :, 1] .= Inintal_condition(model, 0.03 * model.Lx)

    # Fourier transfrom of the initial condition
    uhat = rfft(model.u[:, :, 1])

    for n in 2:length(model.t)
        uhat .= uhat ./ (1 .- model.dt .* model.Lhat)
        
        # Zero-pad uhat to match dimensions of u(x, y, t)
        uhat_full = zeros(Complex{Float64}, model.Nx, model.Ny)
        uhat_full[1:size(uhat, 1), :] .= uhat

        model.u[:, :, n] .= real(rifft(uhat_full))
    end
end

# Creating animation
function animate(model::AdvectionDiffusion)
    anim = @animate for n in 1:length(model.t)
        contourf(model.x, model.y, model.u[:, :, n], level = 50)
        xlabel("x")
        ylabel("y")
        title("Time: $(round(model.t[n], digits = 2))")
        colorbar()
    end
    return anim
end

# Main script in file for running code
model = AdvectionDiffusion() 
solve!(model)
anim = animate(model)
gif(anim, "advection_diffusion.gif", fps = 15)


