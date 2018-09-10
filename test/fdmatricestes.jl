module FDMatricesTest
using Test
using LinearAlgebra
using Statistics
using FDMatrices
using FDMatrices: ownreal

A₁ = [-2 1 1 ; 1 -2 1; 1 1 -2]
A₂ = [-2 1; 1 -2]
F = Matrix(I, 3, 2)
f = reshape(F, 6)




@testset "Combine Kronecker sum and PeriodicMatrix" begin
Aₚ = laplaceperiodic(2, 3)
A = Aₚ⊕Aₚ
Fp = Matrix(I, 3, 3)
fp = reshape(Fp, 9)

@test A₁*Fp + Fp*A₁ ≈ reshape(A*fp, size(Fp))
Am = Matrix(A)
# Aₚ does not have full rank but for x with A(A\v)...
x = Real.(A\fp)
fx = A*x
# it should hold that
@test A\fx ≈ x
@test pinv(A)*fx ≈ pinv(Am)*fx
end;


@testset "Combine Kronecker sum and PeriodicMatrix" begin
Aₚ = laplaceperiodic(2, 3)
A = Aₚ⊕Aₚ
Fp = Matrix(I, 3, 3)
fp = reshape(Fp, 9)

@test A₁*Fp + Fp*A₁ ≈ reshape(A*fp, size(Fp))
Am = Matrix(A)
# Aₚ does not have full rank but for x with A(A\v)...
x = Real.(A\fp)
fx = A*x
# it should hold that
@test A\fx ≈ x
@test pinv(A)*fx ≈ pinv(Am)*fx
end;

@testset "Solver tests" begin
N = 256
Δx = 2π/N
x = [i*Δx for i∈0:N-1]
fn = sin.(x)
g = -sin.(x)
@test solvepoissonperiodic2nd1D(g, Δx) ≈ fn atol=Δx^2
@test solvepoissonperiodic4th1D(g, Δx) ≈ fn atol=Δx^4
@test solvepoissondirichlet2nd1D(g[2:end], 0, 0, Δx) ≈ fn[2:end] atol=Δx^2
@test solvepoissondirichlet4th1D(g[2:end], 0, 0, Δx) ≈ fn[2:end] atol=Δx^4

a = 2
b = 1
fc = x.*((b-a)/2π) .+ a
@test fc[1] == a
@test solvepoissondirichlet2nd1D(g[2:end], a, b, Δx) ≈ fn[2:end]+fc[2:end] atol=Δx^2
@test solvepoissondirichlet4th1D(g[2:end], a, b, Δx) ≈ fn[2:end]+fc[2:end] atol=Δx^4

Δt = 0.01
ft = fn.*exp(-Δt)
@test solveimplicitdiffusionperiodic2nd1D(fn, Δx, Δt) ≈ ft atol=max(Δt^2, Δx^2)
@test solveimplicitdiffusionperiodic4th1D(fn, Δx, Δt) ≈ ft atol=max(Δt^2, Δx^4)
@test solveimplicitdiffusiondirichlet2nd1D(fn[2:end], 0, 0, Δx, Δt) ≈ ft[2:end] atol=max(Δt^2, Δx^2)
@test solveimplicitdiffusiondirichlet4th1D(fn[2:end], 0, 0, Δx, Δt) ≈ ft[2:end] atol=max(Δt^2, Δx^4)

@test solveimplicitdiffusiondirichlet2nd1D(fn[2:end]+fc[2:end], a, b, Δx, Δt) ≈ ft[2:end]+fc[2:end] atol=max(Δt^2, Δx^2)
@test solveimplicitdiffusiondirichlet4th1D(fn[2:end]+fc[2:end], a, b, Δx, Δt) ≈ ft[2:end]+fc[2:end] atol=max(Δt^2, Δx^4)

#f(x,y) = sin(x) + cos(2y)
#g(x,y) = Δf = -sin(x) - 4cos(2y)
Δy = 2π/(N/2)
h = 1
y = [i*Δy for i∈0:(N/2)-1]
f2d = fn .+ (cos.(h*y))'
g2d = g .- (h^2*cos.(h*y))'
@test all(isapprox.(solvepoissonperiodic2nd2D(g2d, [Δx, Δy]), f2d, atol=(h^4+1)*Δy^2))
@test all(isapprox.(solvepoissonperiodic4th2D(g2d, [Δx, Δy]), f2d, atol=(h^6+1)*Δy^4))
end;

@testset "Profiling tests" begin
N = 16
Aₚ = laplaceperiodic(2, N)
Amₚ = Matrix(Aₚ)
# Aₚ does not have full rank but for x with A(A\v)...
v = rand(N)
x = Real.(Aₚ\v)
v = Real.(Aₚ*x)
# it should hold that
@test Aₚ\v ≈ x
@test pinv(Aₚ)*v≈ pinv(Amₚ)*v
#@test Aₚ\v < pinv(Amₚ)*v
@test (@elapsed Aₚ\v) < (@elapsed pinv(Amₚ)*v)

A = Aₚ⊕Aₚ
Fp = Matrix(I, N, N)
fp = reshape(Fp, N^2)

Am = convert(Matrix, A)
# Aₚ does not have full rank but for x with A(A\v)...
x = ownreal.(A\fp)
fx = A*x
# it should hold that
@test A\fx ≈ x
@test pinv(A)*fx ≈ pinv(Am)*fx
@test real(A\fx).-mean(real(A\fx)) ≈ Am\fx.-mean(Am\fx)

@test 10*(@elapsed A\fx) < (@elapsed Am\fx)
end;



end
