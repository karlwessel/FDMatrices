
struct DSTIMatrix <: TransformMatrix{Real}
    N::Int
    debug::Bool
end

struct InvDSTIMatrix <: TransformMatrix{Real}
    S::DSTIMatrix
end

DSTIMatrix(N) = DSTIMatrix(N, false)
InvDSTIMatrix(N::Int, debug=false) = InvDSTIMatrix(DSTIMatrix(N, false))

getm(S::DSTIMatrix) = (N=len(S); [sin(π/(N+1)*n*k) for k∈1:N, n∈1:N])
getm(S::InvDSTIMatrix) = 2/(len(S)+1) * getm(S.S)
len(S::DSTIMatrix) = S.N
len(S::InvDSTIMatrix) = len(S.S)

inv(S::DSTIMatrix) = InvDSTIMatrix(S)
inv(S::InvDSTIMatrix) = S.S

op(S::DSTIMatrix) = FSTI
op(S::InvDSTIMatrix) = iFSTI
