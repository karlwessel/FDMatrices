using FDMatrices: FastMatrix
using LinearAlgebra
import Base: size, *, /, inv, transpose
struct FourierMatrixPlan{T} <: FastMatrix{T}
    plan::AbstractFFTs.Plan{T}
    debug::Bool
end

fouriermatrix(N, ::Type{T}, debug=false) where {T} =
    FourierMatrixPlan(plan_fft(Array{T}(undef, N)), debug)

fouriermatrix(v::AbstractArray{T,N}, debug=false) where {T, N} =
    fouriermatrix(size(v,1), T, debug)

len(A::FourierMatrixPlan) = length(A.plan)
size(A::FourierMatrixPlan) = (len(A), len(A))
getm(A::FourierMatrixPlan{T}) where {T} = A*Matrix{T}(I, size(A))

*(A::FourierMatrixPlan, v::AbstractVector) = A.plan*v
*(A::FourierMatrixPlan, B::AbstractMatrix) = mapslices(x->A*x, B, dims=1)

*(A::AbstractMatrix, B::FourierMatrixPlan) = transpose(B*transpose(A))

inv(B::FourierMatrixPlan) = FourierMatrixPlan(inv(B.plan), B.debug)

function *(A::FourierMatrixPlan, B::FourierMatrixPlan)
    if A.plan == inv(B.plan) || B.plan == inv(A.plan)
        I
    else
        A*getm(B)
    end
end

transpose(A::FourierMatrixPlan) = A

*(A::FourierMatrixPlan, b::Number) =
    FourierMatrixPlan(AbstractFFTs.ScaledPlan(A.plan, b), A.debug)
*(b::Number, A::FourierMatrixPlan) = A*b
/(A::FourierMatrixPlan, b::Number) = A*(1/b)

/(A::AbstractMatrix, B::FourierMatrixPlan) = A*inv(B)
