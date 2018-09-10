module PeriodicMatrixTests
using Test
using LinearAlgebra
using Revise
using FDMatrices
using FDMatrices:getm

@testset "PeriodicMatrix tests" begin
A = PeriodicMatrix([-2, 1, 1], true)
Am = [-2 1 1; 1 -2 1; 1 1 -2]
@test size(A) == (3,3)
@test getm(A) ≈ Am
v = rand(3)
@test A * v ≈ Am * v
@test getm(-A) ≈ -Am
@test getm(A*5) ≈ Am*5
@test getm(A/5) ≈ Am/5
@test getm(2I - A) ≈ 2I - Am
@test getm(transpose(A)) ≈ transpose(Am)
# A does not have full rank but for x with A(A\v)...
x = A\v
# it should hold that
@test A\(A*x) ≈ x

@test sort(eigvals(A)) ≈ eigvals(Am)
@test pinv(A) ≈ pinv(Am)


B = rand(3, 6)
@test A * B ≈ Am * B
# A does not have full rank but for x with A(A\v)...
x = A\B
# it should hold that
@test A\(A*x) ≈ x
end;


end
