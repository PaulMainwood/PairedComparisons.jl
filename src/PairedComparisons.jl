module PairedComparisons

using DataFrames

include("elo.jl")
include("helper.jl")

export Elo, fit!, one_ahead!, predict

end # module
