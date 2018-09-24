using FDMatrices: FastMatrix
using LinearAlgebra
import Base: size, *, /, inv, transpose
struct FourierMatrixPlan{T} <: FastMatrix{T}
    plan::AbstractFFTs.Plan{T}
    debug::Bool
end

fouriermatrix(N::Int, ::Type{T}, debug=false) where {T} =
    FourierMatrixPlan(plan_fft(Array{T}(undef, N)), debug)

fouriermatrix(v::AbstractArray{T,1}, debug=false) where {T} =
    FourierMatrixPlan(plan_fft(v), debug)

len(A::FourierMatrixPlan) = prod(size(A.plan))
size(A::FourierMatrixPlan) = (len(A), len(A))
getm(A::FourierMatrixPlan{T}) where {T} = A*Matrix{T}(I, size(A))

*(A::FourierMatrixPlan, v::AbstractVector) = A.plan*v
*(A::FourierMatrixPlan, B::AbstractMatrix) = mapslices(x->A*x, B, dims=1)
*(A::FourierMatrixPlan, B::Diagonal) = mapslices(x->A*x, B, dims=1)
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

/(A::AbstractVector, B::FourierMatrixPlan) = A*inv(B)
\(B::FourierMatrixPlan, A::AbstractVector) = inv(B)*A

/(A::AbstractMatrix, B::FourierMatrixPlan) = A*inv(B)
\(B::FourierMatrixPlan, A::AbstractMatrix) = inv(B)*A




struct FourierMatrixPlan2d{T} <: FastMatrix{T}
    plan::AbstractFFTs.Plan{T}
    debug::Bool
end

fouriermatrix(N::Int, M::Int, ::Type{T}, debug=false) where {T} =
    FourierMatrixPlan2d(plan_fft(Array{T}(undef, (N, M))), debug)

fouriermatrix(v::AbstractArray{T,2}, debug=false) where {T} =
    FourierMatrixPlan2d(plan_fft(v), debug)

len(A::FourierMatrixPlan2d) = prod(size(A.plan))
size(A::FourierMatrixPlan2d) = (len(A), len(A))
getm(A::FourierMatrixPlan2d{T}) where {T} = A*Matrix{T}(I, size(A))

transpose(A::FourierMatrixPlan2d) = A

*(A::FourierMatrixPlan2d, v::AbstractVector) = reshape(A.plan*reshape(v, size(A.plan)), len(A))
*(A::FourierMatrixPlan2d, B::AbstractMatrix) = mapslices(x->A*x, B, dims=1)
*(A::FourierMatrixPlan2d, B::Diagonal) = mapslices(x->A*x, B, dims=1)
*(A::AbstractMatrix, B::FourierMatrixPlan2d) = transpose(B*transpose(A))

inv(B::FourierMatrixPlan2d) = FourierMatrixPlan2d(inv(B.plan), B.debug)

function *(A::FourierMatrixPlan2d, B::FourierMatrixPlan2d)
    if A.plan == inv(B.plan) || B.plan == inv(A.plan)
        I
    else
        A*getm(B)
    end
end



*(A::FourierMatrixPlan2d, b::Number) =
    FourierMatrixPlan2d(AbstractFFTs.ScaledPlan(A.plan, b), A.debug)
*(b::Number, A::FourierMatrixPlan2d) = A*b
/(A::FourierMatrixPlan2d, b::Number) = A*(1/b)

/(A::AbstractVector, B::FourierMatrixPlan2d) = A*inv(B)
\(B::FourierMatrixPlan2d, A::AbstractVector) = inv(B)*A

/(A::AbstractMatrix, B::FourierMatrixPlan2d) = A*inv(B)
\(B::FourierMatrixPlan2d, A::AbstractMatrix) = inv(B)*A


lazykron(A::FourierMatrixPlan{T}, B::FourierMatrixPlan{T}) where {T} =
    fouriermatrix(len(B), len(A), T, A.debug || B.debug)
