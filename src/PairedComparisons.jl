module PairedComparisons

using DataFrames

include("helper.jl")
include("elo.jl")
include("glicko.jl")

export Elo, Glicko, fit!, one_ahead!, predict

end # module
