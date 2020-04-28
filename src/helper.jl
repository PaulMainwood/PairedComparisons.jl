
#Helper functions for ratings RatingSystems - duplicating, getting games already played etc

function dupe_for_rating(games)
    #Takes dataframe of games and predictions, reverses the players, result and prediction and appends them to end
    original = rename(games, [:P1, :P2, :P1_wins, :P2_wins, :Day], makeunique=true)
    dupe_backwards = rename(games, [:P2, :P1, :P2_wins, :P1_wins, :Day], makeunique=true)
    return sort!(vcat(original, dupe_backwards), :Day)
end

function brier(predictions)
    withoutmissing = filter(!isnan, collect(skipmissing(predictions)))
    return mean((1.0 .- withoutmissing).^2)
end

function cumcount(m)
    d = Dict()
    function g(a)
        d[a] = get(d, a, 0) + 1
    end
    return g.(m)
end

function cumcount(m, surface)
    d = Dict()
    for i in 1:maximum(surface)
        d[i] = Dict()
    end

    function g(a, surface)
        d[surface][a] = get(d[surface], a, 0) + 1
    end

    return g.(m, surface)
end

function games_previously_played_count(training_games, testing_games)
    m = hcat(vcat(training_games[!, 1], testing_games[!, 1]), vcat(training_games[!, 2], testing_games[!, 2]))
    flattened = collect(Iterators.flatten(transpose(m)))
    counts = cumcount(flattened)
    return transpose(reshape(counts, 2, :))
end

function filter_today_played_before(training_games, today_names, threshold)
    #Filter predictions for day on threshold of games played in training set
    relevant = games_previously_played_count(training_games, today_names)[size(training_games)[1] + 1: size(training_games)[1] + size(today_names)[1],:]

    today_names.P1_before = relevant[:, 1]
    today_names.P2_before = relevant[:, 2]

    return filter(row -> row.P1_before >= threshold && row.P2_before >= threshold, today_names)
end

function filter_played_before(predictions, threshold, training_games, testing_games)
    played = games_previously_played_count(training_games, testing_games)
    playedf = minimum(played, dims = 2)
    matchlength = playedf[end - length(predictions) + 1: end]
    return predictions[matchlength .>= threshold]
end

function recent_players(games::DataFrame, recent_days::Int)
    #Return players who have had a game in the last few days
    latest_day = maximum(games.Day)
    games_played_recently = filter(row -> row.Day >= latest_day - recent_days, games)
    return unique(vcat(games_played_recently.P1, games_played_recently.P2))
end

function log_loss(predictions)
    withoutmissing = filter(!isnan, collect(skipmissing(predictions)))
    ll = -sum([log(prediction) for prediction in withoutmissing]) / length(withoutmissing)
    return ll
end

function log_loss(predictions, lower)
    tally = 0
    log_total = 0.0
    for row in 1:length(predictions[1])
        if (predictions[2][row] >= lower) & (predictions[3][row] >= lower) & (!ismissing(predictions[1][row])) & (!isnan(predictions[1][row]))
            tally += 1
            log_total += log(predictions[1][row])
        end
    end
    ll = - log_total / tally
    return ll, tally
end

function add_blanks(playerdayratings::Dict)
    days = collect(keys(sort(playerdayratings)))
    first_day = minimum(days)
    last_day = maximum(days)
    all_days = fill(NaN, last_day - first_day + 1)
    for j in days
        all_days[j - first_day + 1] = playerdayratings[j]
    end
    return all_days
end

function order_by_recency(games::DataFrame)
    P1 = vcat(games.P1)
    P2 = vcat(games.P2)
    return reverse(unique(transpose(hcat(P1, P2))))
end
