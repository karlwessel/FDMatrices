

struct FastKron{T<:AbstractMatrix, S<:AbstractMatrix, V} <: FastMatrix{V}
    A::T
    B::S
    debug::Bool
end
function FastKron(A::T, B::S, debug=false) where {T<:AbstractMatrix, S<:AbstractMatrix}
    FastKron{T,S, promote_type(eltype(A), eltype(B))}(A, B, debug)
end

fastkron(A,B, debug=false) = FastKron(A,B, debug)



function getindex(K::FastKron, x::Int,y::Int)
    if debug(K)
        throw(ErrorException("unexp convert"))
    end
    i1, i2 = divrem(x-1, size(K.A,1))
    j1, j2 = divrem(y-1, size(K.A,2))
    K.A[i1+1,j1+1]*K.B[i2+1,j2+1]
end

getm(K::FastKron) = kron(K.A, K.B)

size(K::FastKron) = size(K.A).*size(K.B)

adjoint(K::FastKron) = fastkron(adjoint(K.A), adjoint(K.B), K.debug)
transpose(K::FastKron) = fastkron(transpose(K.A), transpose(K.B), K.debug)
tr(K::FastKron) = tr(K.A)*tr(K.B)

inv(K::FastKron) = fastkron(inv(K.A), inv(K.B), K.debug)

# function *(K::FastKron, v::AbstractVector)
#     # (A⊗B)v = g <=> BVAᵀ = G
#     # with v = flatten(V), g = flatten(G)
#     V = reshape(v, (size(K.B, 2), size(K.A, 2)))
#     Aᵀ = transpose(K.A)
#     B = K.B
#     reshape(B*V*Aᵀ, size(K.A, 1)*size(K.B, 1))
# end

*(K::FastKron, v::AbstractVector) =
    mul!(similar(v, size(K.B, 1)*size(K.A, 1)), K, v)

function mul!(v2::AbstractVector, K::FastKron, v::AbstractVector)
    # (A⊗B)v = g <=> BVAᵀ = G
    # with v = flatten(V), g = flatten(G)
    V = reshape(v, (size(K.B, 2), size(K.A, 2)))
    V2 = reshape(v2, (size(K.B, 1), size(K.A, 1)))
    Aᵀ = transpose(K.A)
    B = K.B
    mul!(V2,B*V,Aᵀ)
    #mul!(V2,B, V2)
    reshape(V2, size(K.A, 1)*size(K.B, 1))
end


function *(K::FastKron, v::AbstractSparseVector)
    V = reshape(v, (size(K.B, 2), size(K.A, 2)))
    reshape((K.B*V)*transpose(K.A), size(K.A, 1)*size(K.B, 1))
end

\(K::FastKron, A::AbstractVecOrMat) = inv(K)*A

/(A::AbstractVecOrMat, K::FastKron) = A*inv(K)

function kronmul(K::FastKron, A::AbstractMatrix{T}) where {T}
    B = Matrix{T}(undef, size(A))
    @views for i in 1:size(A,2)
        mul!(B[:,i], K, A[:,i])
    end
    B
end
*(K::FastKron, A::AbstractMatrix) = kronmul(K, A)


*(K::FastKron, A::Diagonal) = kronmul(K, A)

*(A::AbstractMatrix, K::FastKron) = transpose(transpose(K)*transpose(A))
*(A::Diagonal, K::FastKron) = transpose(transpose(K)*transpose(A))

+(K1::FastKron, K2::FastKron) = Matrix(K1)+Matrix(K2)
