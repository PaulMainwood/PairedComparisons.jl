module PairedComparisons

using DataFrames

include("helper.jl")
include("elo.jl")
include("glicko.jl")
include("bradleyterry.jl")
include("WHR.jl")
include("oneahead.jl")

export Elo, Glicko, BradleyTerry, BT, WHR, fit!, one_ahead!, predict, iterate!, add_games!, log_likelihood, brier, brierss, actual_scores, log_loss, rating, filter_played_before, display_ratings, change_in_surface, jumble, dupe_for_rating, predict_or_missing

end # module
