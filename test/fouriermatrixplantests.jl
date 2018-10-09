module FourierMatrixPlanTests

using Test
using LinearAlgebra
using FFTW
using Revise
using FDMatrices

using FDMatrices: getm, fouriermatrix

@testset "FourierMatrixPlan tests" begin
F = fouriermatrix(3, Float64, false)
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
end;

@testset "FourierMatrixPlan2d tests" begin
F = fouriermatrix(3,2, Float64, true)
@test size(F) == (6,6)
Fm = getm(F)
@test size(Fm) == size(F)
V = rand(3,2)
v = reshape(V, length(V))

@test F*v ≈ reshape(fft(V), 6)
@test F*v ≈ Fm*v
@test getm(transpose(F)) == Fm
@test getm(inv(F)) ≈ inv(Fm)


V2 = rand(6,6)

@test F*V2 ≈ Fm*V2
@test V2*F ≈ V2*Fm
@test V2/F ≈ V2/Fm

@test F*transpose(V2) ≈ Fm*transpose(V2)
@test transpose(V2)*F ≈ transpose(V2) * Fm
D = Diagonal(v)
@test F*D ≈ Fm*D
end;


@testset "Inverse FourierMatrixPlan2d tests" begin
F = inv(fouriermatrix(3, 2, Float64, true))
Fm = getm(F)
V = rand(3,2)
v = reshape(V, length(V))

@test size(F) == (6,6)
@test F*v ≈ reshape(ifft(V), 6)
@test F*v ≈ Fm*v
@test getm(transpose(F)) == Fm
@test getm(inv(F)) ≈ inv(Fm)

V2 = rand(6,6)

@test F*V2 ≈ Fm*V2
@test V2*F ≈ V2*Fm
@test V2/F ≈ V2/Fm
@test F*transpose(V2) ≈ Fm*transpose(V2)
@test transpose(V2)*F ≈ transpose(V2) * Fm

D = Diagonal(v)
@test F*D ≈ Fm*D
end;

@testset "kronecker product of two fourier matrices" begin
F1 = fouriermatrix(3, Float64, false)
F2 = fouriermatrix(2, Float64, false)

F = lazykron(F1, F2)
Fm = kron(getm(F1), getm(F2))

v = rand(6)

@test F*v ≈ Fm*v
F2 = fastkron(F1, F2)
@test F2*v ≈ Fm*v
end


@testset "FourierMatrixPlan3d tests" begin
F = fouriermatrix(3,2,4, Float64, true)
@test size(F) == (24,24)
Fm = getm(F)
@test size(Fm) == size(F)
V = rand(3,2,4)
v = reshape(V, length(V))

@test F*v ≈ reshape(fft(V), 24)
@test F*v ≈ Fm*v
@test getm(transpose(F)) == Fm
@test getm(inv(F)) ≈ inv(Fm)


V2 = rand(24,24)

@test F*V2 ≈ Fm*V2
@test V2*F ≈ V2*Fm
@test V2/F ≈ V2/Fm

@test F*transpose(V2) ≈ Fm*transpose(V2)
@test transpose(V2)*F ≈ transpose(V2) * Fm
D = Diagonal(v)
@test F*D ≈ Fm*D
end;


@testset "kronecker product of three fourier matrices" begin
F1 = fouriermatrix(3, Float64, true)
F2 = fouriermatrix(2, Float64, true)
F3 = fouriermatrix(4, Float64, true)

F = lazykron(F1, F2)
Fm = kron(getm(F1), getm(F2))

F = lazykron(F, F3)
Fm = kron(Fm, getm(F3))

v = rand(24)

@test F*v ≈ Fm*v
Ff = fastkron(F1, F2)
Ff = fastkron(Ff, F3)
@test Ff*v ≈ Fm*v

Ft = lazykron(F2, F3)
Fmt = kron(getm(F2), getm(F3))

Ft = lazykron(F1, Ft)
Fmt = kron(getm(F1), Fmt)

v = rand(24)

@test Ft*v ≈ Fmt*v

Ftf = fastkron(F2, F3)
Ftf = fastkron(F1, Ftf)
@test Ftf*v ≈ Fmt*v
end
end
