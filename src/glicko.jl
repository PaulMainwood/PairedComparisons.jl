struct Glicko
    c::Float64
    default_rating::Tuple{Float64, Float64}
    ratings::Dict
end

function Glicko(;c = 85.0, default_rating = (1500.0, 350.0), ratings = Dict{Int64, Tuple}())
    #Set up a new Elo struct, either using key words or defaults if not provided
    return Glicko(c, default_rating, ratings)
end

#Fit the Glicko rating to some new data
function fit!(glicko::Glicko, original_games::DataFrame)

    games = original_games[!, (1:5)]

    #Duplicate whole rating dataframe with reversed results
    games = dupe_for_rating(games)

    #Split into days and deal with all players in them separately
    for day_games in groupby(games, :Day)
        #Add glicko predictions to each day
        day_games = DataFrame(day_games)

        #Update the players who are not playing that day
        not_playing = setdiff(keys(glicko.ratings), day_games[!, 1])
        update_not_played!.(Ref(glicko), not_playing)

        #Groupby Player1 including results
        for player_day_games in groupby(day_games, :P1)
            player = player_day_games[1, 1]
            glicko.ratings[player] = update_player_rating(player, player_day_games[!, 2], player_day_games[!, 3], player_day_games[!, 4], glicko)
        end
    end
end

function update_not_played!(glicko, player)
    glicko.ratings[player] = (glicko.ratings[player][1], minimum((sqrt(glicko.ratings[player][2]^2 + glicko.c^2), glicko.default_rating[2])))
end

#g function from Glicko paper
function g(RD::Float64)
    return 1 / sqrt(1 + (0.0000100725 * RD^2))
end

#E function from Glicko paper
function E(r::Float64, rj::Float64, RDj::Float64)
    return 1 / (1 + 10^(g(RDj) * (rj - r) / 400.0))
end

#Reciprocal of d2 function from Glicko paper
function recipd2(r, rj, RDj)
    return 0.000033173 * sum(g.(RDj).^2 .* E.(r, rj, RDj) .* (1 .- E.(r, rj, RDj)))
end

function update_player_rating(player::Int64, opponents, player_wins, opponent_wins, glicko)

    #Collect a tuple of the player being updated...
    (r, RD) = get(glicko.ratings, player, glicko.default_rating)
    #And a Tuple of the ratings and RDs for their opponents
    (rj, RDj) = collect(zip([get(glicko.ratings, opponent, glicko.default_rating) for opponent in opponents]...))

    r_updated = r + ((0.0057565 / (1 / RD^2 + recipd2(r, rj, RDj))) * sum(g.(RDj) .* (player_wins .- ((player_wins + opponent_wins) .* E.(r, rj, RDj)))))
    RD_updated = sqrt(1/((1 / RD^2) + recipd2(r, rj, RDj)))

    return r_updated, RD_updated
end

#Returns the predicted result for any two players in the existing Elo rating dictionary
function predict(m::Glicko, i::Integer, j::Integer)
    r, RD = get(m.ratings, i, m.default_rating)
    rj, RDj = get(m.ratings, j, m.default_rating)
    return 1 / (1 + 10^(g(sqrt(RDj^2+RD^2)) * (rj - r) / 400.0))
end
