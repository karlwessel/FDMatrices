struct KronSum{T} <: FastMatrix{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
    debug::Bool
end
KronSum(A, B) = KronSum(A, B, false)

function getm(v::KronSum)
    Matrix(Matrix(I, size(v.B))⊗v.A + v.B⊗Matrix(I, size(v.A)))
end

function getindex(K::KronSum, x::Int,y::Int)
    if debug(K)
        throw(ErrorException("unexp convert"))
    end
    #println("bla")
    i1, i2 = divrem(x-1, size(K.A,1))
    j1, j2 = divrem(y-1, size(K.A,2))
    a = i1==j1 ? K.A[i2+1,j2+1] : 0

    b = i2==j2 ? K.B[i1+1,j1+1] : 0
    a+b
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

*(A::KronSum, b::Number) = KronSum(A.A*b, A.B*b)
*(b::Number, A::KronSum) = KronSum(b*A.A, b*A.B)
/(A::KronSum, b::Number) = A*inv(b)

+(A::KronSum, B::UniformScaling) = KronSum(A.A+B, A.B)
+(A::UniformScaling, B::KronSum) = B+A
-(A::KronSum, B::UniformScaling) = A+(-B)
-(A::UniformScaling, B::KronSum) = A+(-B)
-(A::KronSum) = KronSum(-A.A, -A.B)

pinv(A::KronSum) = pinv(eigen(A))

\(A::KronSum, B::AbstractVector) = solve(A, B)

function solve(A::KronSum, v::AbstractVector)
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
