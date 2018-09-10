module DirichletMatrixTests
using Test
using LinearAlgebra
using Revise
using FDMatrices
using FDMatrices:getm, dirichleteigenvalues

@testset "DirichletMatrix tests" begin
A = DirichletMatrix([-2, 1, 0], 0, 0, true)
Am = [-2 1 0; 1 -2 1; 0 1 -2]
c = [0 for i∈1:3]

v = rand(3)
@test A * v ≈ Am * v+c
@test (-A)*v ≈ (-Am)*v+c
@test (A*5)*v ≈ (Am*5)*v+c
@test (A/5)*v ≈ (Am/5)*v+c
@test (2I - A)*v ≈ (2I - Am)*v+c
# A does not have full rank but for x with A(A\v)...
x = A\v
# it should hold that
@test A\(A*x) ≈ x

@test sort(dirichleteigenvalues(A, 3)) ≈ eigvals(Am)

end;


end
