module FDMatrices

using LinearAlgebra
using FFTW
using SparseArrays

import Base:*, /, \, +, -
import Base:transpose, inv, convert, size, getindex, promote_rule, isapprox,
    adjoint, setindex!, reverse

import LinearAlgebra: eigen, pinv, eigvals, eigvecs, Matrix, tr

export laplaceperiodic, laplacedirichlet, ⊕, KronSum,
    FourierMatrix, InvFourierMatrix, ⊗, lazykron,
    PeriodicMatrix, DirichletMatrix,
    solvepoissonperiodic2nd1D,
    solvepoissonperiodic4th1D,
    solvepoissondirichlet2nd1D,
    solvepoissondirichlet4th1D,
    solveimplicitdiffusionperiodic2nd1D,
    solveimplicitdiffusionperiodic4th1D,
    solveimplicitdiffusiondirichlet2nd1D,
    solveimplicitdiffusiondirichlet4th1D,
    solvepoissonperiodic2nd2D,
    solvepoissonperiodic4th2D,
    FSTI, iFSTI, DCTI, DSTIMatrix, InvDSTIMatrix,
    PureDirichletMatrix,
    poisson1dtest, poisson1derror,
    poisson2dperiodictest,
    poisson1dperiodictest,
    diff3dtest,
    fouriermatrix



include("utils.jl")
include("LazyKron.jl")
include("FourierMatrix.jl")
include("FourierMatrixPlan.jl")
include("DSTIMatrix.jl")
include("PeriodicMatrix.jl")
include("KronSum.jl")
include("DirichletMatrix.jl")
include("PureDirichletMatrix.jl")
include("testfunctions.jl")

⊗ = lazykron
⊕ = kronsum


function laplaceperiodic(o, N)
    if o == 2
        PeriodicMatrix(a_periodic2ndorder(N))
    elseif o == 4
        PeriodicMatrix(a_periodic4thorder(N))
    end
end

function laplacedirichlet(o, left, right)
    if o == 2
        UniformDirichletMatrix(a_dirichlet2ndorder(2), left, right)
    elseif o == 4
        UniformDirichletMatrix(a_dirichlet4thorder(3), left, right)
    end
end

function a_dirichlet2ndorder(a::Array{T, 1}) where{T}
    a_dirichlet2ndorder(length(a))
end

function a_dirichlet4thorder(a::Array{T, 1}) where{T}
    a_dirichlet4thorder(length(a))
end

function a_dirichlet2ndorder(N::Int)
    a = zeros(N)
    a[1:2] = [-2,1]
    a
end

function a_dirichlet4thorder(N::Int)
    a = zeros(N)
    a[1:3] = [-5//2, 4//3, -1//12]
    a
end

function a_periodic2ndorder(N)
    a = a_dirichlet2ndorder(N)
    a[end] = 1
    a
end

function a_periodic4thorder(N)
    a = a_dirichlet4thorder(N)
    a[end-1:end] = [-1//12, 4//3]
    a
end




"""
    solvepoissonperiodic2nd1D(g, Δx)

Solve the Poisson equation Δf = g for f given the passed vector g and the
passed sampling distance Δx.

Assuming periodic boundary conditions f[end+1] = f[1].
The Laplace operator is approximated with accuracy of second order by
    Δf ≈ Af.
Where
    A = 1/Δx²*[-2  1  0 ... 0 1;
                1 -2  1 0 ... 0;
                0  1 -2 1 0 ...;
                ...].

# Examples

```jldoctest
julia> using FDMatrices

julia> N = 128;

julia> Δx = 2π/N;

julia> f = [sin(Δx*i) for i∈0:N-1];

julia> g = -f;

julia> f2 = solvepoissonperiodic2nd1D(g, Δx);

julia> maximum(abs.(f2 - f)) < Δx^2
true
```
"""
function solvepoissonperiodic2nd1D(g, Δx)
    A = laplaceperiodic(2, g) / Δx^2
    A\g
end

function solvepoissonperiodic2nd2D(g, Δx)
    Ap1 = laplaceperiodic(2, size(g, 1)) / Δx[1]^2
    Ap2 = laplaceperiodic(2, size(g, 2)) / Δx[2]^2

    A = Ap1 ⊕ Ap2

    reshape(A\reshape(g, prod(size(g))), size(g))
end



"""
    solvepoissonperiodic4th1D(g, Δx)

Solve the Poisson equation Δf = g for f given the passed vector g and the
passed sampling distance Δx.

Assuming periodic boundary conditions f[end+1] = f[1].
The Laplace operator is approximated with accuracy of fourth order by
    Δf ≈ Af.
Where
    A = 1/Δx²*[  -5/2   4/3 -1/12     0   ...     0 -1/12   4/3;
                  4/3  -5/2   4/3 -1/12     0   ...     0 -1/12;
                -1/12   4/3  -5/2   4/3 -1/12     0   ...     0;
                    0 -1/12   4/3  -5/2   4/3 -1/12     0   ...;
                ...].

# Examples

```
julia> N = 128
julia> Δx = 2π/N
julia> f = [sin(Δx*i) for i∈0:N-1]
julia> g = -f
julia> maximum(abs.(solvepoissonperiodic4th1D(g, Δx) - f)) < Δx^4
```
"""
function solvepoissonperiodic4th1D(g, Δx)
    A = laplaceperiodic(4, g) / Δx^2
    A\g
end

function solvepoissonperiodic4th2D(g, Δx)
    Ap1 = laplaceperiodic(4, size(g, 1)) / Δx[1]^2
    Ap2 = laplaceperiodic(4, size(g, 2)) / Δx[2]^2

    A = Ap1 ⊕ Ap2

    reshape(A\reshape(g, prod(size(g))), size(g))
end


"""
    solvepoissondirichlet2nd1D(g, left, right, Δx)

Solve the Poisson equation Δf = g for f given the passed vector g, the boundary
conditions at the left and right and the passed sampling distance Δx.

Assuming periodic dirichlet conditions f[0] = left, f[end+1] = right.
The Laplace operator is approximated with accuracy of second order by
    Δf ≈ Af.
Where
    A = 1/Δx²*[-2  1  0 0 0 ...;
                1 -2  1 0 0 ...;
                0  1 -2 1 0 ...;
                ...].

# Examples

```
julia> N = 128
julia> Δx = 2π/N
julia> f = [sin(Δx*i) + 2.0i/N + 1.0 for i∈1:N-1]
julia> g = [-sin(Δx*i) for i∈1:N-1]
julia> maximum(abs.(solvepoissondirichlet2nd1D(g, 1, 3, Δx) - f)) < Δx^2
```
"""
function solvepoissondirichlet2nd1D(g, left, right, Δx)
    A = laplacedirichlet(2, left, right) / Δx^2
    A\g
end

function solvepoissondirichlet2nd2D(g, left, right, Δx)
    Ap1 = laplacedirichlet(2, left, right) / Δx[1]^2
    Ap2 = laplacedirichlet(2, left, right) / Δx[2]^2

    A = Ap1 ⊕ Ap2

    reshape(A\reshape(g, prod(size(g))), size(g))
end

"""
    solvepoissondirichlet4th1D(g, left, right, Δx)

Solve the Poisson equation Δf = g for f given the passed vector g, the boundary
conditions at the left and right and the passed sampling distance Δx.

Assuming periodic dirichlet conditions f[0] = left, f[end+1] = right.
The Laplace operator is approximated with accuracy of fourth order by
    Δf ≈ Af.
Where
A = 1/Δx²*[  -5/2+1/12   4/3 -1/12     0     0     0 ...;
                   4/3  -5/2   4/3 -1/12     0     0 ...;
                 -1/12   4/3  -5/2   4/3 -1/12     0 ...;
                     0 -1/12   4/3  -5/2   4/3 -1/12 ...;
            ...].

# Examples

```
julia> N = 128
julia> Δx = 2π/N
julia> f = [sin(Δx*i) + 2.0i/N + 1.0 for i∈1:N-1]
julia> g = [-sin(Δx*i) for i∈1:N-1]
julia> maximum(abs.(solvepoissondirichlet4th1D(g, 1, 3, Δx) - f)) < Δx^2
```
"""
function solvepoissondirichlet4th1D(g, left, right, Δx)
    A = laplacedirichlet(4, left, right) / Δx^2
    A\g
end



"""
    solveimplicitdiffusion(A, fᵗ)

Solve the implicit timestep of the diffusion equation (I-ΔtΔ)fᵗ⁺¹ = fᵗ for fᵗ⁺¹
given the current solution fᵗ and the approximation A for the laplace operator.

The Laplace Operator is approximated with accuracy of second order
    Δf ≈ Af.

The derivative in time is approximated with Crank-Nicholson implizit time step
with accuracy of second order
    ∂ₜf ≈ fᵗ⁺¹ - fᵗ = 0.5Δt * A(fᵗ⁺¹ + fᵗ)

This gives the equation
    (2I - ΔtA)fᵗ⁺¹ = (2I - ΔtA)fᵗ
to solve.
"""
function solveimplicitdiffusion(A, fᵗ)
    (2I-A)\((2I+A)*fᵗ)
end

"""
    solveimplicitdiffusionperiodic2nd1D(fᵗ, Δx, Δt)

Solve the implicit timestep of the diffusion equation (I-ΔtΔ)fᵗ⁺¹ = fᵗ for fᵗ⁺¹
given the current solution fᵗ, the passed sampling distance Δx and the time
step size Δt.

Assuming periodic boundary conditions f[end+1] = f[1].
The Laplace Operator is approximated with accuracy of second order
    Δf ≈ Af.
Where
    A = 1/Δx²*[-2  1  0 ... 0 1;
                1 -2  1 0 ... 0;
                0  1 -2 1 0 ...;
                ...].

The derivative in time is approximated with Crank-Nicholson implizit time step
with accuracy of second order
    ∂ₜf ≈ fᵗ⁺¹ - fᵗ = 0.5Δt * A(fᵗ⁺¹ + fᵗ)

This gives the equation
    (2I - ΔtA)fᵗ⁺¹ = (2I - ΔtA)fᵗ
to solve.
"""
function solveimplicitdiffusionperiodic2nd1D(fᵗ, Δx, Δt)
    A = laplaceperiodic(2, fᵗ) * Δt / Δx^2

    solveimplicitdiffusion(A, fᵗ)
end

"""
    solveimplicitdiffusiondirichlet2nd1D(fᵗ, left, right, Δx, Δt)

Solve the implicit timestep of the diffusion equation ∂ₜf = Δf.
given the current solution fᵗ, the passed sampling distance Δx and the time
step size Δt for dirichlet boundary conditions:
f[0] = left
f[end+1] = right

The Laplace Operator is approximated with accuracy of second order
    Δf ≈ Af.
Where
    A = 1/Δx²*[-2  1  0 ... 0 0;
                1 -2  1 0 ... 0;
                0  1 -2 1 0 ...;
                ...].

The derivative in time is approximated with Crank-Nicholson implizit time step
with accuracy of second order
    ∂ₜf ≈ fᵗ⁺¹ - fᵗ = 0.5Δt * A(fᵗ⁺¹ + fᵗ)

This gives the equation
    (2I - ΔtA)fᵗ⁺¹ = (2I - ΔtA)fᵗ
to solve.
"""
function solveimplicitdiffusiondirichlet2nd1D(fᵗ, left, right, Δx, Δt)
    A = laplacedirichlet(2, left, right) * Δt / Δx^2

    solveimplicitdiffusion(A, fᵗ)
end


"""
    solveimplicitdiffusionperiodic4th1D(fᵗ, Δx, Δt)

Solve the implicit timestep of the diffusion equation (I-ΔtΔ)fᵗ⁺¹ = fᵗ for fᵗ⁺¹
given the current solution fᵗ, the passed sampling distance Δx and the time
step size Δt.

Assuming periodic boundary conditions f[end+1] = f[1].
The Laplace Operator is approximated with accuracy of fourth order
    Δf ≈ Af.
Where
    A = 1/Δx²*[  -5/2   4/3 -1/12     0   ...     0 -1/12   4/3;
                  4/3  -5/2   4/3 -1/12     0   ...     0 -1/12;
                -1/12   4/3  -5/2   4/3 -1/12     0   ...     0;
                    0 -1/12   4/3  -5/2   4/3 -1/12     0   ...;
                ...].

The derivative in time is approximated with Crank-Nicholson implizit time step
with accuracy of second order
    ∂ₜf ≈ fᵗ⁺¹ - fᵗ = 0.5Δt * A(fᵗ⁺¹ + fᵗ)

This gives the equation
    (2I - ΔtA)fᵗ⁺¹ = (2I - ΔtA)fᵗ
to solve.
"""
function solveimplicitdiffusionperiodic4th1D(fᵗ, Δx, Δt)
    A = laplaceperiodic(4, fᵗ) * Δt / Δx^2

    solveimplicitdiffusion(A, fᵗ)
end

"""
    solveimplicitdiffusiondirichlet4th1D(fᵗ, left, right, Δx, Δt)

Solve the implicit timestep of the diffusion equation ∂ₜf = Δf.
given the current solution fᵗ, the passed sampling distance Δx and the time
step size Δt for dirichlet boundary conditions:
f[0] = left
f[end+1] = right

The Laplace Operator is approximated with accuracy of fourth order
    Δf ≈ Af.
Where
    A = 1/Δx²*[  -5/2+1/12   4/3 -1/12     0     0     0 ...;
                       4/3  -5/2   4/3 -1/12     0     0 ...;
                     -1/12   4/3  -5/2   4/3 -1/12     0 ...;
                         0 -1/12   4/3  -5/2   4/3 -1/12 ...;
                ...].

The derivative in time is approximated with Crank-Nicholson implizit time step
with accuracy of second order
    ∂ₜf ≈ fᵗ⁺¹ - fᵗ = 0.5Δt * A(fᵗ⁺¹ + fᵗ)

This gives the equation
    (2I - ΔtA)fᵗ⁺¹ = (2I - ΔtA)fᵗ
to solve.
"""
function solveimplicitdiffusiondirichlet4th1D(fᵗ, left, right, Δx, Δt)
    A = laplacedirichlet(4, left, right) * Δt / Δx^2

    solveimplicitdiffusion(A, fᵗ)
end

end
