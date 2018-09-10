module FourierMatrixPlanTests

using Test
using LinearAlgebra
using FFTW
using Revise
using FDMatrices

using FDMatrices: getm, fouriermatrix

@testset "FourierMatrixPlan tests" begin
F = fouriermatrix(3, Float64, true)
Fm = getm(F)
v = [1, 2, 3]

@test size(F) == (3,3)
@test F*v ≈ fft(v)
@test F*v ≈ Fm*v
@test getm(transpose(F)) == Fm
@test getm(inv(F)) ≈ inv(Fm)

F2 = fouriermatrix(4, Float64, true)
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


@testset "Inverse FourierMatrixPlan tests" begin
F = inv(fouriermatrix(3, Float64, true))
Fm = getm(F)
v = [1, 2, 3]

@test size(F) == (3,3)
@test F*v ≈ ifft(v)
@test F*v ≈ Fm*v
@test getm(transpose(F)) == Fm
@test getm(inv(F)) ≈ inv(Fm)

F2 = inv(fouriermatrix(4, Float64, true))
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
