struct KronSum{T} <: FastMatrix{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
    debug::Bool
end
KronSum(A, B) = KronSum(A, B, false)

function getm(v::KronSum)
    Matrix(Matrix(I, size(v.B))⊗v.A + v.B⊗Matrix(I, size(v.A)))
end


function *(A::KronSum, b::AbstractVector)
    N, M = size(A.A)
    K, L = size(A.B)
    if N==M && K==L
        B = reshape(b, (N, K))
        return reshape(A.A*B+B*transpose(A.B), N*K)
    else
        return Matrix(A)*b
    end
end

pinv(A::KronSum) = pinv(eigen(A))

\(A::KronSum, B::AbstractVector) = solve(A, B)

function solve(A::KronSum, v::Vector)
    T = eigvecs(A)
    T⁻¹ = inv(T)
    D⁻¹ = pinv(Diagonal(eigvals(A)))
    #Ax = TDT⁻¹x = v => T⁻¹x = D⁻¹T⁻¹v
    T*(D⁻¹*(T⁻¹*v))
end

function eigvals(A::KronSum)
    vals = [a+b for a∈eigvals(A.A), b∈eigvals(A.B)]
    reshape(vals, prod(size(vals)))
end

eigvecs(A::KronSum) = eigvecs(A.B) ⊗ eigvecs(A.A)
eigen(A::KronSum) = Eigen(eigvals(A), eigvecs(A))

size(v::KronSum) = size(v.A).*size(v.B)

kronsum = KronSum




convert(::Type{Matrix}, v::KronSum) = Matrix(v)
