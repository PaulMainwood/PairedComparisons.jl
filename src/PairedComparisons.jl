module PairedComparisons

using DataFrames

include("helper.jl")
include("elo.jl")
include("glicko.jl")
include("bradleyterry.jl")
include("WHR.jl")
include("oneahead.jl")

export Elo, Glicko, BradleyTerry, BT, WHR, fit!, one_ahead!, predict, iterate!, add_games!, loglikelihood, brier, log_loss

end # module
