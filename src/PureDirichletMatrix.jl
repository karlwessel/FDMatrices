struct PureDirichletMatrix{T} <: FastMatrix{T}
    P::PeriodicMatrix{T}
end

debug(D::PureDirichletMatrix) = debug(D.P)

PureDirichletMatrix(row, debug=false) = PureDirichletMatrix(PeriodicMatrix(row, debug))

len(D::PureDirichletMatrix) = len(D.P)

function eigvals(D::PureDirichletMatrix)
    N = len(D)
    ap = spzeros(N+2)
    ap[1:N] .= D.P.row
    DCTI(ap)[2:2+N-1]
end

eigvecs(D::PureDirichletMatrix) = DSTIMatrix(len(D))
eigen(A::PureDirichletMatrix) = Eigen(eigvals(A), eigvecs(A))

pinv(A::PureDirichletMatrix) = pinv(eigen(A))

*(A::PureDirichletMatrix, v::Vector) = iFSTI(eigvals(A) .* FSTI(v))
*(A::PureDirichletMatrix, a::Number) = PureDirichletMatrix(A.P*a)
/(A::PureDirichletMatrix, a::Number) = PureDirichletMatrix(A.P/a)

\(A::PureDirichletMatrix, v::Vector) = iFSTI(calcfprime(FSTI(v), eigvals(A)))
-(A::PureDirichletMatrix) = PureDirichletMatrix(-A.P)

+(A::UniformScaling, B::PureDirichletMatrix) = PureDirichletMatrix(A+B.P)

-(A::UniformScaling, B::PureDirichletMatrix) = A + (-B)
transpose(A::PureDirichletMatrix) = A

getm(A::PureDirichletMatrix) = eigvecs(A) * Diagonal(eigvals(A)) / eigvecs(A)
size(A::PureDirichletMatrix) = size(A.P)
