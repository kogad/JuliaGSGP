using JuliaGSGP
using Test

import JuliaGSGP: GPNode, PrimitiveFunction, Variable, Const, evaluate, Dataset
import JuliaGSGP: Entry
import JuliaGSGP: optimal_coeff, projection_crossover_entry


tests = ["evaluate", "prgp"]

if length(ARGS) > 0
    tests = ARGS
end

@testset "JuliaGSGP" begin

for t in tests
    fp = joinpath(dirname(@__FILE__), "test_$t.jl")
    println("$fp ...")
    include(fp)
end

end
