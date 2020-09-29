##############################################################################################################################################
# Implements the Elo algorithm as published by Arpad Elo in the 1960s. 
# Two differences from the standard implementations. first, the scores are done with natural powers rather than the common Elo style of base
# ten and a scaling of 400 points. To get back, you use.
#
# Natural = Elo_style * ln(10) / 400
#
# In addition, Elo are traditionally started at 1000, where in this case, we set zero as the default rating (so negative scores are possible.)
# 
# The second difference is the implementation of time periods, which matches Elo's original approach - re-rating each time period or tournament
# rather than treating each game separately. You can recover the more common game-by-game approach by simply omitting the time period column.
# This algorithm is also much faster (fast_fit!, below)
##############################################################################################################################################

#Struct is an Elo object - in this version it's a dict and a base kfactor
struct Elo
    kfac::Real
    ratings::Dict
    default_rating::Real
end

function Elo(;kfac = 0.31, ratings = Dict{Int64, Float64}(), default_rating = 0.0)
    #Set up a new Elo struct, either using key words or defaults if not provided
    return Elo(kfac, ratings, default_rating)
end


function fit!(m::Elo, games::DataFrame; verbose = false)
    #= Wrapper function to choose the right approach to fit a dataframe of games =#
    titles = names(games)

    if !issubset(["P1", "P2", "P1_wins", "P2_wins"], titles)
        error("For rating, we need at least four columns named P1, P2, P1_wins, P2_wins")
    end

    #Depending on whether time periods are specified or not
    if !in("Period", names(games))
        verbose && println("No distinct periods found. Treating each game as a separate rating period and using fastelo algorithm.")
        fast_fit!.(Ref(m), games.P1, games.P2, games.P1_wins, games.P2_wins)
    else
        period_fit!(m, games[!, [:P1, :P2, :P1_wins, :P2_wins, :Period]])
    end
    return nothing
end


function fast_fit!(m::Elo, P1, P2, P1_wins, P2_wins)
    #=
    This function is used when matches themsleves define time periods. It
    simply updates the two players' elo ratings once per match
    Do not use when one is doing tournament or by-day elo updates where one might
    have many-one/one-many matches.
    =#
    surprise = m.kfac * (P1_wins - predict(m, P1, P2) * (P1_wins + P2_wins))
    m.ratings[P1] = get(m.ratings, P1, m.default_rating) + surprise
    m.ratings[P2] = get(m.ratings, P2, m.default_rating) - surprise
end


function period_fit!(m::Elo, original_games::DataFrame)
    #=
    Proper fit function, using defined periods to re-rate in. Takes only 
    columns of a dataframe named (P1, P2, P1_wins, P2_wins, Day)
    =#
    #Duplicate whole rating dataframe with reversed results
    games = dupe_for_rating(original_games)

    #Split into days
    for day_games in groupby(games, :Period, sort = true)
        #Add elo predictions to each game in each day
        day_games = DataFrame(day_games)
        day_games[!, :Predict] = predict.(Ref(m), day_games.P1, day_games.P2) .* (day_games.P1_wins .+ day_games.P2_wins) 
        #Groupby Player1 including results
        aggregated_by_P1_games = combine(groupby(day_games, :P1), :P1_wins => sum => :P1_wins, :P2_wins => sum => :P2_wins, :Predict => sum => :Predict)
        update!(m, aggregated_by_P1_games)
    end
end

function predict(m::Elo, i::Integer, j::Integer; rating_day::Integer = 0)
    #Returns the predicted result for any two players in the existing Elo rating dictionary, for a single game (%)
    return 1.0 / (1.0 + exp((get(m.ratings, j, m.default_rating) - get(m.ratings, i, m.default_rating))))
end

function update!(m::Elo, games)
    #Update function allowing for period in which there may be duplicate games
    for row in eachrow(games)
        if ismissing(row.P1_wins)
            return nothing
        end
        m.ratings[row.P1] = get(m.ratings, row.P1, m.default_rating) + m.kfac * (row.P1_wins - row.Predict)
    end
end

function display_ratings(elo::Elo)
    a = collect(keys(elo.ratings))
    b = map.(x -> elo.ratings[x], a)
    players = DataFrame(player_code = a, player_rating = b)
    return sort!(players, :player_rating, rev = true)
end

function display_ratings(elo::Elo, players)
    player_dict = Dict(players[:, 1] .=> players[:, 2])
    ratings = display_ratings(elo)
    ratings.player_names = map.(x -> player_dict[x], ratings.player_code)
    return ratings
end
