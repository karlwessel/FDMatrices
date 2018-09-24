module LazyKronTests

using Test
using LinearAlgebra
using SparseArrays
using Revise
using FDMatrices
using FDMatrices:getm

@testset "LazyKron tests: general" begin
A = [1 2 3; 4 5 6]
B = [7 1; 8 1; 9 1; 10 1]

K = lazykron(A, B, true)
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

K2 = lazykron(rand(2,2), rand(3,3), true)
K2m = getm(K2)
@test getm(inv(K2)) ≈ inv(K2m)
@test tr(K2) ≈ tr(K2m)

Ks = lazykron(A, B)
Ksm = getm(Ks)
@test Ks+Ks ≈ Ksm+Ksm
end;

@testset "LazyKron tests: vectors" begin
A = [1 2 3; 4 5 6]
B = [7 1; 8 1; 9 1; 10 1]

K = lazykron(A, B, true)
Km = getm(K)

v = rand(6)
@test K*v ≈ Km*v
end;

@testset "LazyKron tests: SparseVectors" begin
A = [1 2 3; 4 5 6]
B = [7 1; 8 1; 9 1; 10 1]

K = lazykron(A, B, true)
Km = getm(K)

vsp = spzeros(6)
vsp[2] = 5
vsp[6] = -2
@test K*vsp ≈ Km*vsp
end;

@testset "LazyKron tests: matrizes" begin
K2 = lazykron(rand(2,2), rand(3,3), true)
K2m = getm(K2)

M = rand(6,6)
@test K2*M ≈ K2m*M
@test K2\M ≈ K2m\M
@test M/K2 ≈ M/K2m
end;

@testset "LazyKron tests: sparse matrizes" begin
K2 = lazykron(rand(2,2), rand(3,3), true)
K2m = getm(K2)

Ms = spzeros(6,6)
Ms[2,2] = 5
Ms[4,6] = -2

@test K2*Ms ≈ K2m*Matrix(Ms)
@test K2\Ms ≈ K2m\Matrix(Ms)
@test Ms/K2 ≈ Matrix(Ms)/K2m
end;

@testset "LazyKron tests: Transpose matrizes" begin
K2 = lazykron(rand(2,2), rand(3,3), true)
K2m = getm(K2)

M = transpose(rand(6,6))

@test K2*M ≈ K2m*M
@test K2\M ≈ K2m\M
@test M/K2 ≈ M/K2m
end;

@testset "LazyKron tests: Adjoint matrizes" begin
K2 = lazykron(rand(2,2), rand(3,3), true)
K2m = getm(K2)

M = adjoint(rand(6,6))

@test K2*M ≈ K2m*M
@test K2\M ≈ K2m\M
@test M/K2 ≈ M/K2m
end;

@testset "LazyKron tests: Diagonal matrizes" begin
K2 = lazykron(rand(2,2), rand(3,3), true)
K2m = getm(K2)

M = Diagonal(rand(6))

@test K2*M ≈ K2m*M
@test K2\M ≈ K2m\M
@test M/K2 ≈ M/K2m
end;

end
