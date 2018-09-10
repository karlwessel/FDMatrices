struct UniformDirichletMatrix{T}
    row::SparseVector{T}
    left
    right
    debug::Bool
end
DirichletMatrix = UniformDirichletMatrix
UniformDirichletMatrix(row, left, right) = UniformDirichletMatrix(row, left, right, false)



function UniformDirichletMatrix(v::Array, left, right, debug=false)
    UniformDirichletMatrix(SparseVector(v), left, right, debug)
end



function row(A::UniformDirichletMatrix, N)
    M = maximum(A.row.nzind)
    if M > N
        throw(ErrorException("Dirichlet matrix order is to large."))
    end

    a = spzeros(N)
    a[1:M] = A.row[1:M]
    a
end

function dirichleteigenvalues(A::UniformDirichletMatrix, N)
    ap = row(A, N+2)
    DCTI(ap)[2:2+N-1]
end

function bounds(A::UniformDirichletMatrix, N)
    a = row(A, N+1)[2:N+1]
    a[1] += 2*a[2]
    a*A.left + a[end:-1:1]*A.right
end

function *(A::UniformDirichletMatrix, v::AbstractVector)
    iFSTI(dirichleteigenvalues(A, length(v)) .* FSTI(v)) + bounds(A, length(v))
end

function *(A::UniformDirichletMatrix, v::Number)
    UniformDirichletMatrix(A.row*v, A.left, A.right)
end

function /(A::UniformDirichletMatrix, v::Number)
    UniformDirichletMatrix(A.row/v, A.left, A.right)
end

function \(A::UniformDirichletMatrix, v::AbstractVector)
    c = bounds(A, length(v))
    iFSTI(calcfprime(FSTI(v-c), dirichleteigenvalues(A, length(v))))
end

function -(A::UniformDirichletMatrix)
    UniformDirichletMatrix(-A.row, A.left, A.right)
end

function +(A::UniformScaling, B::UniformDirichletMatrix)
    tmp = B.row
    tmp[1] += A.λ
    UniformDirichletMatrix(tmp, B.left, B.right)
end

function -(A::UniformScaling, B::UniformDirichletMatrix)
    A + (-B)
end
