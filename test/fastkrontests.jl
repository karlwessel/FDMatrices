module FastKronTests

using Test
using LinearAlgebra
using SparseArrays
using Revise
using FDMatrices
using FDMatrices:getm

for T in [Float64, ComplexF64]
@testset "FastKron $T test" begin
@testset "FastKron tests: general" begin
A = T[1 2 3; 4 5 6]
B = [7 1; 8 1; 9 1; 10 1]

K = fastkron(A, B, true)
@test size(K) == (8, 6)
Km = getm(K)
@test Km == kron(A,B)

v = rand(6)
@test K*v ≈ Km*v
@test getm(adjoint(K)) ≈ adjoint(Km)
@test getm(transpose(K)) ≈ transpose(Km)

vsp = spzeros(6)
vsp[2] = 5
vsp[6] = -2
@test K*vsp ≈ Km*vsp

K2 = fastkron(rand(2,2), rand(3,3), true)
K2m = getm(K2)
@test getm(inv(K2)) ≈ inv(K2m)
@test tr(K2) ≈ tr(K2m)

Ks = fastkron(A, B)
Ksm = getm(Ks)
@test Ks+Ks ≈ Ksm+Ksm
end;

@testset "FastKron tests: vectors" begin
A = T[1 2 3; 4 5 6]
B = [7 1; 8 1; 9 1; 10 1]

K = fastkron(A, B, true)
Km = getm(K)

v = rand(6)
@test K*v ≈ Km*v
end;

@testset "FastKron tests: SparseVectors" begin
A = T[1 2 3; 4 5 6]
B = [7 1; 8 1; 9 1; 10 1]

K = fastkron(A, B, true)
Km = getm(K)

vsp = spzeros(6)
vsp[2] = 5
vsp[6] = -2
@test K*vsp ≈ Km*vsp
end;

@testset "FastKron tests: matrizes" begin
K2 = fastkron(rand(T, 2, 2), rand(3, 3), true)
K2m = getm(K2)

M = rand(6,6)
@test K2*M ≈ K2m*M
@test K2\M ≈ K2m\M
@test M/K2 ≈ M/K2m
end;

@testset "FastKron tests: sparse matrizes" begin
K2 = fastkron(rand(T,2,2), rand(3,3), true)
K2m = getm(K2)

Ms = spzeros(6,6)
Ms[2,2] = 5
Ms[4,6] = -2

@test K2*Ms ≈ K2m*Matrix(Ms)
@test K2\Ms ≈ K2m\Matrix(Ms)
@test Ms/K2 ≈ Matrix(Ms)/K2m
end;

@testset "FastKron tests: Transpose matrizes" begin
K2 = fastkron(rand(T, 2,2), rand(3,3), true)
K2m = getm(K2)

M = transpose(rand(6,6))

@test K2*M ≈ K2m*M
@test K2\M ≈ K2m\M
@test M/K2 ≈ M/K2m
end;

@testset "FastKron tests: Adjoint matrizes" begin
K2 = fastkron(rand(T, 2,2), rand(3,3), true)
K2m = getm(K2)

M = adjoint(rand(6,6))

@test K2*M ≈ K2m*M
@test K2\M ≈ K2m\M
@test M/K2 ≈ M/K2m
end;

@testset "FastKron tests: Diagonal matrizes" begin
K2 = fastkron(rand(T, 2,2), rand(3,3), true)
K2m = getm(K2)

M = Diagonal(rand(6))

@test K2*M ≈ K2m*M
@test K2\M ≈ K2m\M
@test M/K2 ≈ M/K2m
end;
end
end
end
