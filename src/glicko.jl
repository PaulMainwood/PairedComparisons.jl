##############################################################################################################################################
# Implements the Glicko algorithm from Mark Glickman as outlined in the papers: http://www.glicko.net/glicko/glicko.pdf (short version)
# and http://www.glicko.net/research/glicko.pdf (technical version).
# In both cases, the original papers use an artificial point scale for comparability with the Elo system, which complicates the calaculations
# considerably. In the below, we are using natural ratings, which can be related back to the Elo scale by a simple scaling factor.
#
# Natural = Elo_style * ln(10) / 400
#
# In addition, Elo are traditionally started at 1000, where in this case, we set zero as the default rating (so negative scores are possible.)
# This is in keeping with the rest of this package, including the Elo.
##############################################################################################################################################

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
    games_duped = dupe_for_rating(games)

    #Split into days and deal with all players in them separately
    for day_games in groupby(games_duped, :Period)
        #Add glicko predictions to each day
        day_games = DataFrame(day_games)
        day = day_games.Period[1]

        #Groupby Player1 including results
        for player_day_games in groupby(day_games, :P1)
            player = player_day_games.P1[1]
            glicko.ratings[player] = update_player_rating(glicko, day, player, player_day_games[!, 2], player_day_games[!, 3], player_day_games[!, 4])
        end
    end
end

function RD_drift(RD, c, defaultRD, day, last_rated)
    #Return the drifted value of RD given the time the rating  was last updated
    if last_rated < 0
        return defaultRD
    else
        return min(sqrt(RD^2 + ((c^2) * abs(day - last_rated))), defaultRD)
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

#Returns the predicted result for any two players in the existing Glicko rating dictionary
function predict(m::Glicko, i::Integer, j::Integer; rating_day::Integer = 0, kwargs...)
    r, RD = rating(m, i, rating_day = rating_day)
    rj, RDj = rating(m, j, rating_day = rating_day)
    return 1 / (1 + exp(g(sqrt(RDj^2 + RD^2)) * (rj - r)))
end

#Returns the rating of a player on a day. If the rating day is before the last day rated, we just take their RD from that day
function rating(m::Glicko, i::Int64; rating_day::Integer = 0)
    r, RD, last_rated = get(m.ratings, i, m.default_rating)

    if rating_day > last_rated
        RD = RD_drift(RD, m.c, m.default_rating[2], rating_day, last_rated)
    end

    return r, RD
end

function rating(m::Glicko, i::Int64, j::Int64; rating_day::Integer = 0)
    return rating(m, i, rating_day = rating_day), rating(m, j, rating_day = rating_day)
end

function display_ratings(m::Glicko; rating_day::Integer = 0)
    a = collect(keys(m.ratings))
    b = map.(x -> glicko.ratings[x][1], a)
    c = map.(x -> glicko.ratings[x][2], a)
    d = map.(x -> glicko.ratings[x][3], a)

    if rating_type == "elo"
        b = convert_natural_to_elo.(b)
        c = convert_natural_to_elo.(c)
    end

    players = DataFrame(player_code = a, player_rating = b, player_sd = c, last_day_played = d)
    return sort!(players, :player_rating, rev = true)
end