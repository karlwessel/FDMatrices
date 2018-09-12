
struct PeriodicPoissonTest1D

end

function pure_triangle(x)
    sum([(-1)^i*(2i+1.0)^(-2)*sin((2i+1)*(x-π/2)) for i∈0:10])
end

function pure_triangle_der(x, order = 2)
    @assert mod(order,2)==0
    sum([(-1)^(i+order/2)*(2i+1.0)^(-2+order)*sin((2i+1)*(x-π/2)) for i∈0:10])
end

function triangle(x)
    pure_triangle(x) / -pure_triangle(0)
end

#function triangle_der(x)
#    sum([(-1)^(i+1)*sin((2i+1)*(x-π/2)) for i∈0:10])
#end

function triangle_der(x, order = 2)
    pure_triangle_der(x,order) / -pure_triangle(0)
end

function poisson1dtest(N, shift=0)
    x = [(i-1)/N*2*π for i∈1:N]
    Δx = 2π/N
    f = triangle.(x.+shift)
    g = triangle_der.(x.+shift)

    f,g,Δx,x
end

function poisson2dtest(size, shift=[0,0.2π])
    N, M = size
    x = [(i-1)/N*2*π for i∈1:N]
    y = [(j-1)/M*2*π for j∈1:M]
    Δx = [2π/N, 2π/M]
    fn = (x, y) -> triangle(x+shift[1]) + triangle(y+shift[2])
    gn = (x, y) -> triangle_der(x+shift[1]) + triangle_der(y+shift[2])
    f = fn.(x, y')
    g = gn.(x, y')

    f,g,Δx, x,y
end

poisson2dperiodictest(N) = poisson2dtest(N, [0,0])

poisson1dperiodictest(N) = poisson1dtest(N, 0)
poisson1dhomogentest(N) = poisson1dtest(N, π/2)


function poisson1derror(N, order)
    @assert order in [2,4]
    m = 2+order
    if order == 2
        α = 2
    elseif order == 4
        α = 8
    end
    x = [(i-1)/N*2*π for i∈1:N]
    Δx = 2π/N
    abs.(α/factorial(m)*Δx^order*triangle_der.(x, m) / -triangle(0))
end

function evalpoisson1derror(fn, count=10)
    Ns = [2^i for i∈3:count]
    dxs = zeros(Float64, length(Ns))
    meanerr = zeros(Float64, length(Ns))
    maxerr = zeros(Float64, length(Ns))
    for (i,N) ∈ enumerate(Ns)
        f,g,dx,x = poisson1dtest(N)
        dxs[i] = dx
        err = abs.(fn(g, dx)-f)
        meanerr[i] = mean(err)
        maxerr[i] = maximum(err)
    end
    dxs,meanerr,maxerr
end


function gendiff3dtest(x,y,z,t)
    sin(2π*x)*cos(2π*y)*cos(2π*z)*exp(-12*π^2*t)
end

function diff3dtest(N,M,L, t)
    [gendiff3dtest(x/N,y/M,z/L,t) for x in 0:N-1, y in 0:M-1, z in 0:L-1]
end
diff3dtest(N,M,L) = diff3dtest(N,M,L, 0)
diff3dtest(N) = diff3dtest(N, 0)
diff3dtest(N, t) = diff3dtest(N,N,N, t)
