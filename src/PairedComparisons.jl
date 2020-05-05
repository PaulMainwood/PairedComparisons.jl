module PairedComparisons

using DataFrames

include("helper.jl")
include("elo.jl")
include("glicko.jl")
include("bradleyterry.jl")
include("WHR.jl")

export Elo, Glicko, BradleyTerry, BT, WHR, fit!, one_ahead!, predict, iterate!, add_games!, loglikelihood

end # module
