module DSTIMatrixTests

using Test
using LinearAlgebra
using FFTW
using Revise
using FDMatrices

using FDMatrices: getm

@testset "DSTIMatrix tests" begin
F = DSTIMatrix(3, true)
Fm = getm(F)
v = [1, 2, 3]

@test size(F) == (3,3)
@test F*v ≈ FSTI(v)
@test F*v ≈ Fm*v
@test getm(transpose(F)) == Fm
@test getm(inv(F)) ≈ inv(Fm)

F2 = DSTIMatrix(4, true)
F2m = getm(F2)
V = rand(3,4)

@test F*V*F2 ≈ Fm*V*F2m
@test F*V ≈ Fm*V
@test V*F2 ≈ V*F2m
@test V/F2 ≈ V/F2m
@test F2*transpose(V) ≈ F2m*transpose(V)
@test transpose(V)*F ≈ transpose(V) * Fm

D = Diagonal(v)
@test F*D ≈ Fm*D
@test D*F ≈ D*F
end;


@testset "InvDSTIMatrix tests" begin
F = InvDSTIMatrix(3, true)
Fm = getm(F)
v = [1, 2, 3]

@test size(F) == (3,3)
@test F*v ≈ iFSTI(v)
@test F*v ≈ Fm*v
@test getm(transpose(F)) == Fm
@test getm(inv(F)) ≈ inv(Fm)

F2 = InvDSTIMatrix(4)
F2m = getm(F2)
V = rand(3,4)

@test F*V*F2 ≈ Fm*V*F2m
@test F*V ≈ Fm*V
@test V*F2 ≈ V*F2m
@test V/F2 ≈ V/F2m
@test F2*transpose(V) ≈ F2m*transpose(V)
@test transpose(V)*F ≈ transpose(V) * Fm

D = Diagonal(v)
@test F*D ≈ Fm*D
@test D*F ≈ D*F
end;


end
