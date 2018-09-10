

struct LazyKron{T, S, V} <: FastMatrix{V}
    A::AbstractMatrix{T}
    B::AbstractMatrix{S}
    debug::Bool
end
function LazyKron(A::AbstractMatrix{T}, B::AbstractMatrix{S}, debug=false) where {T, S}
    LazyKron{T,S, promote_type(T, S)}(A, B, debug)
end

lazykron = LazyKron

getm(K::LazyKron) = kron(K.A, K.B)

size(K::LazyKron) = size(K.A).*size(K.B)

adjoint(K::LazyKron) = lazykron(adjoint(K.A), adjoint(K.B))
transpose(K::LazyKron) = lazykron(transpose(K.A), transpose(K.B))
tr(K::LazyKron) = tr(K.A)*tr(K.B)

inv(K::LazyKron) = lazykron(inv(K.A), inv(K.B))

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

\(K::LazyKron, A::AbstractMatrix) = inv(K)*A

function *(K::LazyKron, A::AbstractMatrix)
    cat([K*A[:,i] for i∈1:size(A,2)]..., dims=2)
end

function *(K::LazyKron, A::Diagonal)
    cat([K*A[:,i] for i∈1:size(A,2)]..., dims=2)
end

+(K1::LazyKron, K2::LazyKron) = Matrix(K1)+Matrix(K2)
