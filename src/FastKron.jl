

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
    mul!(similar(v, promote_type(eltype(K.A),eltype(K.B), eltype(v)), size(K.B, 1)*size(K.A, 1)), K, v)

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
ldiv!(Y::AbstractVecOrMat, K::FastKron, A::AbstractVecOrMat) = mul!(Y, inv(K), A)

/(A::AbstractVecOrMat, K::FastKron) = A*inv(K)

function kronmul!(B::AbstractMatrix, K::FastKron, A::AbstractMatrix)
    Aᵀ = transpose(K.A)
    V = reshape(A, (size(K.B, 2), size(K.A, 2), size(A,2)))
    V2 = reshape(B, (size(K.B, 1), size(K.A, 1), size(A,2)))
    @views for i in 1:size(A,2)
        mul!(V2[:,:,i],K.B*V[:,:,i],Aᵀ)
        #mul!(B[:,i], K, A[:,i])
    end
    B
end
mul!(B::AbstractMatrix, K::FastKron, A::AbstractMatrix) = kronmul!(B,K,A)
*(K::FastKron, A::AbstractMatrix) = kronmul!(similar(A, promote_type(eltype.([K.A, K.B, A])...)), K, A)


*(K::FastKron, A::Diagonal) = kronmul!(Matrix{promote_type(eltype.([K.A, K.B, A])...)}(A), K, A)

*(A::AbstractMatrix, K::FastKron) = transpose(transpose(K)*transpose(A))
*(A::Diagonal, K::FastKron) = transpose(transpose(K)*transpose(A))

+(K1::FastKron, K2::FastKron) = Matrix(K1)+Matrix(K2)
