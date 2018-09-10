
struct FourierMatrix{T<:Union{Int, AbstractFFTs.Plan}} <: TransformMatrix{Complex}
    planOrN::T
    debug::Bool
end

struct InvFourierMatrix  <: TransformMatrix{Complex}
    N::Int
    debug::Bool
end

FourierMatrix(T) = FourierMatrix(T, false)
InvFourierMatrix(T) = InvFourierMatrix(T, false)


len(A::FourierMatrix{Int}) = A.planOrN
len(A::FourierMatrix{AbstractFFTs.Plan}) = size(A)[1]
len(A::InvFourierMatrix) = A.N

op(A::FourierMatrix) = fft
op(A::InvFourierMatrix) = ifft

function getm(A::FourierMatrix)
    N = len(A)
    w = exp(-2π*im/N)
    [w^(k*l) for k ∈ 0:N-1, l ∈ 0:N-1]
end

function getm(A::InvFourierMatrix)
    w = exp(-2π*im/A.N)
    [conj(w^(k*l))/A.N for k ∈ 0:A.N-1, l ∈ 0:A.N-1]
end

inv(A::FourierMatrix) = InvFourierMatrix(len(A))
inv(A::InvFourierMatrix) = FourierMatrix(A.N)


*(A::FourierMatrix, V::AbstractMatrix, B::FourierMatrix) = fft(V)
*(A::InvFourierMatrix, V::AbstractMatrix, B::InvFourierMatrix) = ifft(V)
