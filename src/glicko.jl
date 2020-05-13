struct Glicko
    c::Float64
    default_rating::Tuple{Float64, Float64, Int64}
    ratings::Dict
end

function Glicko(;c = 0.1, default_rating = (0, 2.0, -1), ratings = Dict{Int64, Tuple}())
    #Set up a new Elo struct, either using key words or defaults if not provided
    return Glicko(c, default_rating, ratings)
end

#Fit the Glicko rating to some new data
function fit!(glicko::Glicko, original_games::DataFrame)

    games = original_games[!, (1:5)]

    #Duplicate whole rating dataframe with reversed results
    games = dupe_for_rating(games)

    #Split into days and deal with all players in them separately
    for day_games in groupby(games, 5)
        #Add glicko predictions to each day
        day_games = DataFrame(day_games)
        day = day_games[1, 5]

        #Groupby Player1 including results
        for player_day_games in groupby(day_games, 1)
            player = player_day_games[1, 1]
            glicko.ratings[player] = update_player_rating(glicko, day, player, player_day_games[!, 2], player_day_games[!, 3], player_day_games[!, 4])
        end
    end
end

function RD_drift(RD, c, defaultRD, day, last_rated)
    #Return the drifted value of RD given the time the rating  was last updated
    if last_rated < 0
        return defaultRD
    else
        return min(sqrt(RD^2 + ((c^2) * (day - last_rated))), defaultRD)
    end
end

#g function from Glicko paper
function g(RD::Float64)
    return 1 / sqrt(1 + (0.304 * RD^2))
end

#E function from Glicko paper
function E(r::Float64, rj::Float64, RDj::Float64)
    return 1 / (1 + exp(g(RDj) * (rj - r)))
end

#Reciprocal of d2 function from Glicko paper
function recipd2(r, rj, RDj)
    return sum(g.(RDj).^2 .* E.(r, rj, RDj) .* (1 .- E.(r, rj, RDj)))
end

function update_player_rating(glicko::Glicko, day::Int64, player::Int64, opponents, player_wins, opponent_wins)

    #Collect a tuple of the player being updated...
    (r, RD, last_rated) = get(glicko.ratings, player, glicko.default_rating)
    #Adjusts current RD based on when last rated
    RD = RD_drift(RD, glicko.c, glicko.default_rating[2], day, last_rated)

    #And a vector of the Tuples of the ratings and RDs for their opponents
    rj, RDj, last_rated_j = collect(zip(get.(Ref(glicko.ratings), opponents, Ref(glicko.default_rating))...))
    #Again, giving drift
    RDj = RD_drift.(RDj, Ref(glicko.c), Ref(glicko.default_rating[2]), Ref(day), last_rated_j)

    r_updated = r + ((1 / (1 / RD^2 + recipd2(r, rj, RDj))) * sum(g.(RDj) .* (player_wins .- ((player_wins + opponent_wins) .* E.(r, rj, RDj)))))
    RD_updated = sqrt(1/((1 / RD^2) + recipd2(r, rj, RDj)))

    return (r_updated, RD_updated, day)
end

#Returns the predicted result for any two players in the existing Elo rating dictionary
function predict(m::Glicko, i::Integer, j::Integer, day::Integer)
    r, RD, last_rated = get(m.ratings, i, m.default_rating)
    rj, RDj, last_rated_j = get(m.ratings, j, m.default_rating)

    if day > last_rated
        RD = RD_drift(RD, m.c, m.default_rating[2], day, last_rated)
    end
    if day > last_rated_j
        RDj = RD_drift(RDj, m.c, m.default_rating[2], day, last_rated_j)
    end
    return 1 / (1 + exp(g(sqrt(RDj^2 + RD^2)) * (rj - r)))
end
