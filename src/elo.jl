#Struct is an Elo object - in this version it's a dict and a base kfactor
struct Elo
    kfac::Real
    ratings::Dict
    default_rating::Real
end

function fit!(m::Elo, games::DataFrame; verbose = false)
    #= Wrapper function to choose the right approach to fit a dataframe of games =#
    if size(games, 2) == 4
        if verbose println("No periods found. Treating each game as a separate rating period and using fastelo algorithm.") end
        fast_fit!.(Ref(m), games[!, 1], games[!, 2], games[!, 3], games[!, 4])
    elseif size(games, 2) >= 5
        period_fit!(m, games[!, (1:5)])
    else throw(DomainError(games, "DataFrame must have at least four columns, P1_ID, P2_ID, P1_wins, P2_wins."))
    end

return m
end


function fast_fit!(m::Elo, P1, P2, P1_won, P2_won)
    #=
    This function is used when matches themsleves define time periods. It
    simply updates the two players' elo ratings once per match

    Do not use when one is doing tournament or by-day elo updates where one might
    have many-one/one-many matches.
    =#
    surprise = m.kfac * (P1_won - predict(m, P1, P2, P1_won, P2_won))
    m.ratings[P1] = get(m.ratings, P1, m.default_rating) + surprise
    m.ratings[P2] = get(m.ratings, P2, m.default_rating) - surprise
end

#Fit the Elo ratings to some new data
function period_fit!(m::Elo, original_games::DataFrame)
    #=
    Proper fit function, using defined periods to re-rate in. Takes only first 5
    columns of a dataframe (P1, P2, P1_wins, P2_wins, Day) and gets rid of everything else.
    =#
    games = original_games[!, (1:5)]
    #Duplicate whole rating dataframe with reversed results
    games = dupe_for_rating(games)

    #Split into days
    for day_games in groupby(games, 5)
        #Add elo predictions to each game in each day
        day_games = DataFrame(day_games)
        day_games[!, :Predict] = predict.(Ref(m), day_games[:, 1], day_games[:, 2], day_games[:, 3], day_games[:, 4])
        #Groupby Player1 including results
        aggregated_by_P1_games = by(day_games, 1, P1_wins = :P1_wins => sum, P2_wins = :P2_wins => sum, Predict = :Predict => sum)
        update!(m, aggregated_by_P1_games)
    end
end


#Predicts with one-ahead for each day
function one_ahead!(m::Elo, games::DataFrame; prediction_function = predict)
    sort!(games, 5)
    predictions = Float64[]
    #Split into days which are then predicted ahead of time
    for day_games in groupby(games, 5)
        day_games = DataFrame(day_games)
        p = prediction_function.(Ref(m), day_games[:, 1], day_games[:, 2])
        predictions = vcat(predictions, p)
        fit!(m, day_games)
    end
    return predictions
end

#Returns the predicted result for any two players in the existing Elo rating dictionary, for a single game (%)
function predict(m::Elo, i::Integer, j::Integer)

    return 1.0 / (1.0 + 10.0 ^ ((get(m.ratings, j, m.default_rating) - get(m.ratings, i, m.default_rating)) / 400.0))
end

#Returns the predicted result for any two players in the existing Elo rating dictionary, for multiple matches
function predict(m::Elo, i::Integer, j::Integer, P1_games, P2_games)
    return (P1_games + P2_games) / (1.0 + 10.0 ^ ((get(m.ratings, j, m.default_rating) - get(m.ratings, i, m.default_rating)) / 400.0))
end

#Update function allowing for period in which there may be duplicate games
function update!(m::Elo, games)
    for row in eachrow(games)
        if ismissing(row.P1_wins)
            return nothing
        end
        m.ratings[row.P1] = get(m.ratings, row.P1, m.default_rating) + m.kfac * (row.P1_wins - row.Predict)
    end
end
