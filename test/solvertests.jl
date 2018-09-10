module SolverTest
using Test
using LinearAlgebra
using Statistics
using FDMatrices
using Base.Iterators: product

poisson(A, g) = try A\g catch; pinv(A)*g end
poissoninh(A, g, left, right) =
    poisson(A, g+A*Vector(range(left, stop=right, length=size(g,1)+2)[2:end-1]))

poisson2d(Ax, Ay, g) = reshape(poisson(Ax⊕Ay, reshape(g, length(g))), size(g))

poisson3d(Ax, Ay, Az, g) = reshape(poisson(Ax⊕Ay⊕Az, reshape(g, length(g))), size(g))

diffusive_crank(A, g) = poisson(2I-A, (2I+A)*g)
diffusive(A,g) = poisson(I-A, g)

diffusive2d(Ax, Ay, g) =  diffusive(Ax⊕Ay, g)

Ad2 = [-2 1 0 0 0; 1 -2 1 0 0; 0 1 -2 1 0; 0 0 1 -2 1; 0 0 0 1 -2]
Ap2 = [-2 1 0 0 1; 1 -2 1 0 0; 0 1 -2 1 0; 0 0 1 -2 1; 1 0 0 1 -2]

Ad4 = [-29  16  -1   0   0;
        16 -30  16  -1   0;
        -1  16 -30  16  -1;
         0  -1  16 -30  16;
         0   0  -1  16 -29]
 Ap4 = [-30  16  -1  -1  16;
         16 -30  16  -1  -1;
         -1  16 -30  16  -1;
         -1  -1  16 -30  16;
         16  -1  -1  16 -30]
debug = true
 Af = [PureDirichletMatrix([-2, 1, 0, 0, 0], debug),
       PeriodicMatrix([-2, 1, 0, 0, 1], debug),
       PureDirichletMatrix([-30, 16, -1, 0, 0], debug),
       PeriodicMatrix([-30, 16, -1, -1, 16], debug)]
names = ["Dirichlet 2nd", "Periodic 2nd", "Dirichlet 4th", "Periodic 4th"]
As = [Ad2, Ap2, Ad4, Ap4]


@testset "Poisson tests" begin
f = rand(5)
for (name, A1, A2)∈zip(names, As, Af)
    @testset "$name" begin
    g = A1*f
    diff = poisson(A1, g) - f
    @test (diff .- diff[1]) ≈ zeros(5) atol=1e-14

    diff = poisson(A1, g) - poisson(A2, g)
    @test (diff .- diff[1]) ≈ zeros(5) atol=1e-14
    end
end

Ainh = [("Dirichlet inh 2nd", Ad2, Af[1]),
        ("Dirichlet inh 4th", Ad4, Af[3])]
a = rand()
b = a + rand()
c = Vector(range(a, stop=b, length=7)[2:end-1])
for (name, A1, A2)∈Ainh
    @testset "$name" begin
    g = A1*f
    @test poissoninh(A1, g, a, b) ≈ f+c
    @test poissoninh(A1, g, a, b) ≈ poissoninh(A2, g, a, b)
    end
end
end

@testset "2D Poisson tests" begin
f = rand(5,5)
combsS = product(As, As)
combsF = product(Af, Af)
combsN = product(names, names)
for (name, (Ax, Ay), (Ax2, Ay2)) in zip(combsN, combsS, combsF)
    @testset "$name" begin
    g = Ax*f + f*Ay
    diff = poisson2d(Ax, Ay, g) - f
    @test maximum(abs.(diff .- diff[1])) ≈ 0 atol=1e-13

    diff = poisson2d(Ax, Ay, g) - poisson2d(Ax2, Ay2, g)
    @test maximum(abs.(diff .- diff[1])) ≈ 0 atol=1e-13
    end
end
end

@testset "3D Poisson tests" begin
f = rand(5,5,5)
combsS3 = product(As, As, As)
combsF3 = product(Af, Af, Af)
combsN3 = product(names, names, names)
for (name, (Ax, Ay, Az), (Ax2, Ay2, Az2)) in zip(combsN3, combsS3, combsF3)
    @testset "$name" begin
    F = reshape(f, (5*5, 5))
    g = reshape((Ax⊕Ay)*F + F*Az, (5,5,5))
    diff = poisson3d(Ax, Ay, Az, g) - f
    @test maximum(abs.(diff .- diff[1])) ≈ 0 atol=1e-13

    diff = poisson3d(Ax, Ay, Az, g) - poisson3d(Ax2, Ay2, Az2, g)
    @test maximum(abs.(diff .- diff[1])) ≈ 0 atol=1e-13
    end
end
end


@testset "Implizit diffusion tests" begin
f = rand(5)
for (name, A1, A2)∈zip(names, As, Af)
    g = (I-A1)*f
    @testset "$name" begin

    diff = diffusive(A1, g) - f
    @test (diff .- diff[1]) ≈ zeros(5) atol=1e-14

    diff = diffusive(A1, g) - diffusive(A2, g)
    @test (diff .- diff[1]) ≈ zeros(5) atol=1e-14
    end

    @testset "$name Crank-Nicholson" begin
    diff = diffusive_crank(A1, g) - diffusive_crank(A2, g)
    @test (diff .- diff[1]) ≈ zeros(5) atol=1e-13
    end
end
end

@testset "Analytic Poisson tests" begin
for N∈[8,16,32,64,128,256,512,1024,2048,4096,8192]
    @testset "N=$N order=2" begin
    a = zeros(Float64, N)
    a[1:2]=[-2,1]
    a[end]=1
    A = PeriodicMatrix(a, true)
    f,g,dx,x = poisson1dtest(N)
    α = 1/dx^2

    f2 = poisson(A*α, g)
    diff = abs.(f2.-mean(f2) - f)
    @test all(diff .< poisson1derror(N,2).+1e-9)
    #@test all(diff .<= poisson1derror(N,2))
    end

    @testset "N=$N order=4" begin
    a = zeros(Float64, N)
    a[1:3]=[-30,16,-1]
    a[end-1:end]=[-1,16]
    A = PeriodicMatrix(a, true)
    f,g,dx,x = poisson1dtest(N)
    α = 1/12/dx^2

    f2 = poisson(A*α, g)
    diff = abs.(f2.-mean(f2) - f)
    @test all(diff .< poisson1derror(N,4).+1e-9)
    #@test all(diff .<= poisson1derror(N,2))
    end
end
end
end
