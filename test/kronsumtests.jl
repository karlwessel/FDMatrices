module KronSumTests
using Test
using LinearAlgebra
using FDMatrices
using FDMatrices: getm

@testset "Kronecker sum tests" begin
A₁ = [-2 1 1 ; 1 -2 1; 1 1 -2]
A₂ = [-2 1; 1 -2]
F = rand(3,2)
A = KronSum(A₁, A₂, true)
f = reshape(F, 6)
@assert size(A) == (6, 6)
@test A₁*F + F*A₂ ≈ reshape(A*f, size(F))
Am = getm(A)

@test typeof(A*f)<:AbstractVector
@test A*f ≈ Am*f
@test pinv(A) ≈ pinv(Am)
# Aₚ does not have full rank but for x with A(A\v)...
x = A\f
# it should hold that
@test A\(A*x) ≈ x
end;

@testset "Kronecker sum tests Complex" begin
A₁ = [-2 1 1 ; 1 -2 1; 1 1 -2].+1im
A₂ = [-2 1; 1 -2].-2im
F = rand(3,2)
A = KronSum(A₁, A₂, true)
f = reshape(F, 6)
@assert size(A) == (6, 6)
@test A₁*F + F*A₂ ≈ reshape(A*f, size(F))
Am = getm(A)

@test typeof(A*f)<:AbstractVector
@test A*f ≈ Am*f
@test pinv(A) ≈ pinv(Am)
# Aₚ does not have full rank but for x with A(A\v)...
x = A\f
# it should hold that
@test A\(A*x) ≈ x

end;
end
