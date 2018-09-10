module Benchmarks
using Test
using FFTW
using Statistics
using BenchmarkTools
using AbstractTrees
using FDMatrices:fouriermatrix, FourierMatrix

AbstractTrees.children(a::BenchmarkGroup) = collect(values(a))

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.01
BenchmarkTools.DEFAULT_PARAMETERS.time_tolerance = 0.5

if isfile("prof.json")
    oldsuite = BenchmarkTools.load("prof.json")
    map(x->println(x), Leaves(oldsuite))
end

skip_matrix = false
suite = BenchmarkGroup()
suite["plan_fft"] = BenchmarkGroup()
suite["FourierPlan"] = BenchmarkGroup()
suite["fft"] = BenchmarkGroup()
suite["Fourier"] = BenchmarkGroup()


for i in 1:1
    N=2^i
    println(" N = $N")

    b = suite
    id = "Matrix $(N)x$(N)"

    v = rand(N, N)
    Ap = fouriermatrix(v)
    A = FourierMatrix(N)
    plan = plan_fft(v, 1)
    @test fft(v, 1) ≈ A*v ≈ Ap*v

    b["plan_fft"][id] = (@benchmark $plan*$v)
    b["FourierPlan"][id] = (@benchmark $Ap*$v)
    b["fft"][id] = (@benchmark fft($v, 1))
    b["Fourier"][id] = (@benchmark $A*$v)
    b["plan_fft"][id] = (@benchmark $plan*$v)

    println("vs plan: $(judge(median(b["FourierPlan"][id]), median(b["plan_fft"][id])))")
    println("vs Fourier: $(judge(median(b["FourierPlan"][id]), median(b["Fourier"][id])))")
end
skip_matrix = false
for i in 1:1
    N=2^i
    println(" N = $N")

    b = suite
    id = "Vector $(N)"

    v = rand(N)
    Ap = fouriermatrix(v)
    A = FourierMatrix(N)
    plan = plan_fft(v)
    @test fft(v) ≈ A*v ≈ Ap*v

    b["plan_fft"][id] = (@benchmark $plan*$v)
    b["FourierPlan"][id] = (@benchmark $Ap*$v)
    b["fft"][id] = (@benchmark fft($v))
    b["Fourier"][id] = (@benchmark $A*$v)
    b["plan_fft"][id] = (@benchmark $plan*$v)

    println("vs plan: $(judge(median(b["FourierPlan"][id]), median(b["plan_fft"][id])))")
    println("vs Fourier: $(judge(median(b["FourierPlan"][id]), median(b["Fourier"][id])))")
end
println(collect(Leaves(median(suite))))
println(collect(Leaves(judge(median(suite), median(suite)))))
# if isfile("prof.json")
#     oldsuite = BenchmarkTools.load("prof.json")
#     println(judge(median(suite), median(oldsuite)))
# end
BenchmarkTools.save("prof.json", suite)
end
