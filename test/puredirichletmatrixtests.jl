module PureDirichletMatrixTests
using Test
using LinearAlgebra
using Revise
using FDMatrices
using FDMatrices:getm

@testset "PureDirichletMatrix tests" begin
A = PureDirichletMatrix([-2, 1, 0], true)
Am = [-2 1 0; 1 -2 1; 0 1 -2]
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


A2 = PureDirichletMatrix([-30, 16, -1, 0, 0], true)
Am2 = [-29 16 -1 0 0; 16 -30 16 -1 0;-1 16 -30 16 -1;
        0 -1 16 -30 16; 0 0 -1 16 -29]
@test size(A2) == (5,5)
@test getm(A2) ≈ Am2
v = rand(5)
@test A2 * v ≈ Am2 * v
@test getm(-A2) ≈ -Am2
@test getm(A2*5) ≈ Am2*5
@test getm(A2/5) ≈ Am2/5
@test getm(2I - A2) ≈ 2I - Am2
@test getm(transpose(A2)) ≈ transpose(Am2)
# A does not have full rank but for x with A(A\v)...
x = A2\v
# it should hold that
@test A2\(A2*x) ≈ x

@test sort(eigvals(A2)) ≈ eigvals(Am2)
@test pinv(A2) ≈ pinv(Am2)
end;


end
