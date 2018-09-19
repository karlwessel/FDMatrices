

struct PeriodicMatrix{T} <: FastMatrix{T}
    row::Array{T}
    debug::Bool
end
PeriodicMatrix(A) = PeriodicMatrix(A, false)
CirculantMatrix = PeriodicMatrix

len(A::PeriodicMatrix) = length(A.row)

eigvals(A::PeriodicMatrix) = real(fft(A.row))
eigvecs(A::PeriodicMatrix) = FourierMatrix(length(A.row))
eigen(A::PeriodicMatrix) = Eigen(eigvals(A), eigvecs(A))

pinv(A::PeriodicMatrix) = pinv(eigen(A))

function *(A::PeriodicMatrix, v::Vector)
    irfft(rfft(A.row) .* rfft(v), length(v))
end

function *(A::PeriodicMatrix, B::AbstractMatrix)
    irfft(rfft(A.row) .* rfft(B, 1), size(B,1), 1)
end

*(A::PeriodicMatrix, v::Number) = PeriodicMatrix(A.row*v)
*(v::Number, A::PeriodicMatrix) = A*v

/(A::PeriodicMatrix, v::Number) = PeriodicMatrix(A.row/v)

function \(A::PeriodicMatrix, v::Vector)
    irfft(pinv(Diagonal(rfft(A.row))) * rfft(v), length(v))
    #irfft(calcfprime(rfft(v), rfft(A.row)), length(v))
end

function \(A::PeriodicMatrix, B::AbstractMatrix)
    irfft(pinv(Diagonal(rfft(A.row))) * rfft(B,1), size(B,1), 1)
    #irfft(calcfprime(rfft(B,1), rfft(A.row)), size(B,1),1)
end

function /(A::AbstractMatrix, B::PeriodicMatrix)
    transpose(irfft(rfft(A,2) * pinv(Diagonal(rfft(B.row))), size(A,2), 2))
    #transpose(irfft(calcfprime(rfft(A,2), rfft(B.row)), size(A,2),2))
end

-(A::PeriodicMatrix) = PeriodicMatrix(-A.row)

function +(A::UniformScaling, B::PeriodicMatrix)
    tmp = copy(B.row)
    tmp[1] += A.λ
    PeriodicMatrix(tmp)
end
+(A::PeriodicMatrix, B::UniformScaling) = B + A

-(A::UniformScaling, B::PeriodicMatrix) = A + (-B)
-(A::PeriodicMatrix, B::UniformScaling) = A + (-B)

transpose(A::PeriodicMatrix) = A

function getm(A::PeriodicMatrix)
    N = length(A.row)
    Matrix([A.row[mod(k-i, N)+1] for i∈0:N-1, k∈0:N-1])
end

function getindex(A::PeriodicMatrix, i::Int,k::Int)
    if debug(A)
        throw(ErrorException("unexp convert"))
    end
    N = length(A.row)
    A.row[mod(k-i, N)+1]
end

convert(::Type{Matrix}, A::PeriodicMatrix) = Matrix(A)

size(A::PeriodicMatrix) = (length(A.row), length(A.row))
