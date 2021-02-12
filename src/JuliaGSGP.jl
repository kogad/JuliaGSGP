module JuliaGSGP

using DataStructures
using Random
using Statistics
using LinearAlgebra
using AbstractTrees, Printf

import Base.show

export gsgp, prgp, dgsgp, dgsgp_opt_mate
export ramped_half_and_half, generate_grow, generate_full
export Config, mse, rmse
export PrimitiveSet, add_function!, add_const!, add_variable!, add_rand!
export protected_div, aq
export Record
export reconstruct_tree

include("primitives.jl")
include("config.jl")
include("entry.jl")
include("fitness.jl")
include("record.jl")
include("util.jl")
include("selection.jl")
include("gsgp.jl")
include("dgsgp.jl")
include("prgp.jl")
include("print.jl")

end
