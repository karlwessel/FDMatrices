abstract type FastMatrix{T} <: AbstractMatrix{T} end
debug(F::FastMatrix) = F.debug
Matrix(K::FastMatrix) = !debug(K) ? getm(K) : throw(ErrorException("unexp convert"))

getindex(K::FastMatrix, i) = !debug(K) ? getindex(Matrix(K), i) : throw(ErrorException("unexp convert"))
getindex(K::FastMatrix, I::Vararg{Int, N}) where{N} = !debug(K) ? getindex(Matrix(K), I...) : throw(ErrorException("unexp convert"))

setindex!(K::FastMatrix, v, i) = throw(ErrorException("Assignment to read only matrix!"))

abstract type TransformMatrix{T} <: FastMatrix{T} end

transpose(A::TransformMatrix) = A
size(A::TransformMatrix) = (len(A), len(A))

/(A::AbstractMatrix, B::TransformMatrix) = A*inv(B)

*(A::TransformMatrix, v::Vector) = op(A)(v)

*(A::TransformMatrix, B::AbstractMatrix) = op(A)(B, 1)

*(A::AbstractMatrix, B::TransformMatrix) = op(B)(A, 2)

*(A::TransformMatrix, B::Transpose) = transpose(parent(B)*A)

*(A::Transpose, B::TransformMatrix) = transpose(B*parent(A))

*(A::Diagonal, B::TransformMatrix) = op(B)(A, 2)

*(A::TransformMatrix, B::Diagonal) = op(A)(B, 1)

pinv(A::Eigen) = A.vectors * pinv(Diagonal(A.values)) / A.vectors

"""
    calcfprime(g, a)

Calculate elementwise division of the passed vectors and replace any
NaN in the result with zero.

# Examples

```jldoctext
julia> calcfprime([1,2,3,4], [0,4,0,2])
```
"""
function calcfprime(gp, a)
    r = gp./a
    r[isnan.(r)] .= 0
    r
end



"""
    DSTI(x)

Calculate discrete sine transform of type I of x.

# Examples

```jldoctext
julia> x = collect(1:7)
julia> DSTI(sin.(π/8*x)) ≈ [4, 0, 0, 0, 0, 0, 0]
true
julia> DSTI(sin.(2π/8*x)) ≈ [0, 4, 0, 0, 0, 0, 0]
true
```
"""
function DSTI(x)
    N=length(x)
    T = [sin(π/(N+1)*j*k) for j∈1:N, k∈1:N]
    T*x
end

"""
    iDSTI(x)

Calculate inverse of the discrete sine transform of type I of x.

# Examples

```jldoctext
julia> x = collect(1:7)
julia> iDSTI([4, 0, 0, 0, 0, 0, 0]) ≈ sin.(π/8*x)
true
```
"""
function iDSTI(x)
    DSTI(x)*2/(length(x)+1)
end

"""
    FSTI(x)

Calculate fast (discrete) sine transform of type I of x using the fft.

# Examples

```jldoctext
julia> x = collect(1:7)
julia> FSTI(sin.(π/8*x)) ≈ [4, 0, 0, 0, 0, 0, 0]
true
julia> FSTI(sin.(2π/8*x)) ≈ [0, 4, 0, 0, 0, 0, 0]
true
```
"""
function FSTI(x, dims=[])
    N = size(x)
    if length(dims) == 0
        dims = collect(1:ndims(x))
    end
    xfill = x
    for i∈dims
        D = collect(size(xfill))
        D[i] = 1
        Z = zeros(tuple(D...))
        xfill = cat(Z, xfill, Z, -reverse(xfill, dims=i), dims=i)
    end
    ind = (i∉dims ? (1:N[i]) : (2:N[i]+1) for i∈1:length(N))
    ownreal.((0.5im)^(length(dims))*rfft(xfill, dims))[ind...]
end

ownreal(x) = x
ownreal(z::Complex{T}) where {T} = isapprox(real(z), z, atol=1e-10) ? real(z) : throw(InexactError(Symbol(string(T)), T, z))
function FSTI(x::AbstractArray{T}, dims=[]) where {T<:Complex}
    N = size(x)
    if length(dims) == 0
        dims = collect(1:ndims(x))
    end
    xfill = x
    for i∈dims
        D = collect(size(xfill))
        D[i] = 1
        Z = zeros(tuple(D...))
        xfill = cat(Z, xfill, Z, -reverse(xfill, dims=i), dims=i)
    end
    ind = (i∉dims ? (1:N[i]) : (2:N[i]+1) for i∈1:length(N))
    (0.5im)^(length(dims))*fft(xfill, dims)[ind...]
end

FSTI(x::Diagonal, dims=[]) = FSTI(SparseMatrixCSC(x), dims)

"""
    iFSTI(x)

Calculate inverse of the discrete sine transform of type I of x using the fft.

# Examples

```jldoctext
julia> x = collect(1:7)
julia> iFSTI([4, 0, 0, 0, 0, 0, 0]) ≈ sin.(π/8*x)
true
```
"""
function iFSTI(x, dims=[])
    if length(dims) == 0
        dims = collect(1:ndims(x))
    end
    Z = prod(2/(N+1) for N∈size(x)[dims])
    FSTI(x, dims)*Z
end

"""
    DCTI(x)

Calculate discrete cosine transform of type I of x using the fft.

# Examples

```jldoctext
julia> DCTI(cos.(π*[0//3, 1//3, 2//3, 3//3])) ≈ [0, 3, 0, 0]
true
```
"""
function DCTI(v)
    N = length(v)
    # DCT([abcd]) == DFT([abcdcb])
    vp = [i<=N ? v[i] : v[2N-i] for i∈1:2N-2]
    real(rfft(vp))[1:N]
end


"""
    calcdiag(T, T⁻¹, A)

Calculate the diagonal matrix of the passed Matrix using the passed Transforms.

The transforms should be functions that transform a vector. And for which
there exists a Matrix T such that T(x) = Tx.

The diagonal matrix D = T⁻¹AT is calculated assuming T=Tᵀ is symmetric by
    D = T(T⁻¹A)ᵀ

# Examples

```jldoctext
julia> A = [2 1 0 1; 1 2 1 0; 0 1 2 1; 1 0 1 2]
julia> calcdiag(fft, ifft, A) ≈ Diagonal([4, 2, 0, 2])
true
```
"""
function calcdiag(T,Tinv, A)
    # apply T^-1 to columns of A
    temp = hcat([Tinv(A[:,i]) for i∈1:size(A)[2]]...)
    # apply T to rows of result
    vcat([reshape(T(temp[i,:]),(1,size(A)[1])) for i∈1:size(A)[1]]...)
end

"""
    extractdiag(T, T⁻¹, A)

Calculate the Eigenvalues of the passed matrix using the passed transforms
as Eigenvectors

# Examples

```jldoctext
julia> A = [2 1 0 1; 1 2 1 0; 0 1 2 1; 1 0 1 2]
julia> extractdiag(fft, ifft, A)
```
"""
function extractdiag(T, Tinv, A)
    D = calcdiag(T, Tinv, A)
    @assert D ≈ Diagonal(D)
    diag(D)
end


"""
    solve(A, g, T, T⁻¹)

Solve the linear equation Ax=g for the passed A and g assuming the
passed transforms as Eigenvectors of A.

# Examples

```jldoctext
julia> A = [2 1 0 1; 1 2 1 0; 0 1 2 1; 1 0 1 2]
julia> f = [1, 0, 1, 0]
julia> g = A*f
julia> fp = solve(A, g, fft, ifft)
julia> f ≈ fp
True
```
"""
function solve(A, g, T, Tinv)
    aₖ = extractdiag(T, Tinv, A)
    gp = Tinv(g)
    fp = calcfprime(gp, aₖ)
    T(fp)
end

% = mod
"""
    extractdiag_per(A)

Calculate the Eigenvalues of the passed matrix assuming A is circular.

# Examples

```jldoctext
julia> A = [2 1 0 1; 1 2 1 0; 0 1 2 1; 1 0 1 2]
julia> extractdiag_per(A)
```
"""
function extractdiag_per(A)
    v = A[1,:]
    fft(v)
end



"""
    extractdiag_dir(A, [k=2])

Calculate the Eigenvalues of A assuming DSTI as Eigenvectors.

This corresponds to A defining Dirichlet boundaries.

k defines the row of A from which to calculate the Eigenvalues

# Examples

```jldoctext
julia> A = [2 1 0 0; 1 2 1 0; 0 1 2 1; 0 0 1 2]
julia> extractdiag_dir(A)
```
"""
function extractdiag_dir(A, k=2)
    v = A[k,k:end]
    N = size(A)[1]
    vp = zeros(N+2)
    vp[1:N-k+1] = v
    DCTI(vp)[2:N+2-1]
end

"""
    circulant(a)

Return the circulant matrix defined by having the passed vector as first row.

# Exmaples

```
julia> circulant([2,1,0,1])
4×4 Array{Int64,2}:
 2  1  0  1
 1  2  1  0
 0  1  2  1
 1  0  1  2
```
"""
function circulant(a)
    N = length(a)
    [a[(j-i) % length(a) + 1] for i ∈ 0:(N-1), j ∈ 0:(N-1)]
end

import Base: kron
function kron(A::UniformScaling, B::Array{T,2}) where {T}
    kron(Matrix(A, size(B)), B)
end

function kron(A::Array{T,2}, B::UniformScaling) where {T}
    kron(A, Matrix(B, size(A)))
end
