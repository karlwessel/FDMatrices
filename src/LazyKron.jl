

struct LazyKron{T, S, V} <: FastMatrix{V}
    A::AbstractMatrix{T}
    B::AbstractMatrix{S}
    debug::Bool
end
function LazyKron(A::AbstractMatrix{T}, B::AbstractMatrix{S}, debug=false) where {T, S}
    LazyKron{T,S, promote_type(T, S)}(A, B, debug)
end

lazykron(A,B, debug=false) = LazyKron(A,B, debug)



function getindex(K::LazyKron, x::Int,y::Int)
    if debug(K)
        throw(ErrorException("unexp convert"))
    end
    i1, i2 = divrem(x-1, size(K.A,1))
    j1, j2 = divrem(y-1, size(K.A,2))
    K.A[i1+1,j1+1]*K.B[i2+1,j2+1]
end

getm(K::LazyKron) = kron(K.A, K.B)

size(K::LazyKron) = size(K.A).*size(K.B)

adjoint(K::LazyKron) = lazykron(adjoint(K.A), adjoint(K.B), K.debug)
transpose(K::LazyKron) = lazykron(transpose(K.A), transpose(K.B), K.debug)
tr(K::LazyKron) = tr(K.A)*tr(K.B)

inv(K::LazyKron) = lazykron(inv(K.A), inv(K.B), K.debug)

function *(K::LazyKron, v::AbstractVector)
    # (A⊗B)v = g <=> BVAᵀ = G
    # with v = flatten(V), g = flatten(G)
    V = reshape(v, (size(K.B, 2), size(K.A, 2)))
    Aᵀ = transpose(K.A)
    B = K.B
    reshape(B*V*Aᵀ, size(K.A, 1)*size(K.B, 1))
end

function *(K::LazyKron, v::AbstractSparseVector)
    V = reshape(v, (size(K.B, 2), size(K.A, 2)))
    reshape((K.B*V)*transpose(K.A), size(K.A, 1)*size(K.B, 1))
end

\(K::LazyKron, A::AbstractVecOrMat) = inv(K)*A

/(A::AbstractVecOrMat, K::LazyKron) = A*inv(K)

function kronmul(K::LazyKron, A::AbstractMatrix)
    B = Matrix{eltype(A)}(undef, size(A))
    for i in 1:size(A,2)
        B[:,i] = K*A[:,i]
    end
    B
end
*(K::LazyKron, A::AbstractMatrix) = kronmul(K, A)


*(K::LazyKron, A::Diagonal) = kronmul(K, A)

*(A::AbstractMatrix, K::LazyKron) = transpose(transpose(K)*transpose(A))
*(A::Diagonal, K::LazyKron) = transpose(transpose(K)*transpose(A))

+(K1::LazyKron, K2::LazyKron) = Matrix(K1)+Matrix(K2)
