#Predicts with one-ahead for each day for Elo
function one_ahead!(m::Elo, games_original::DataFrame; prediction_function = predict)
    #Predicts with one-ahead for each day
    #games = sort(games_original, :Period)
    games = games_original
    predictions = Float64[]

    if !in("Handicap", names(games))
        games.Handicap = 0.0
    end

    #Split into days which are then predicted ahead of time
    for day_games in groupby(games, :Period)
        day_games = DataFrame(day_games)
        p = prediction_function.(Ref(m), day_games.P1, day_games.P2, day_games.Handicap)
        predictions = vcat(predictions, p)
        fit!(m, day_games)
    end
    return predictions
end

#Predicts with one-ahead for each day for Glicko
function one_ahead!(m::Glicko, games::DataFrame; prediction_function = predict)
    #sort!(games, :Period)
    predictions = Float64[]
    #Split into days which are then predicted ahead of time
    for day_games in groupby(games, :Period)
        day_games = DataFrame(day_games)
        p = prediction_function.(Ref(m), day_games.P1, day_games.P2, day_games.Period[1])
        predictions = vcat(predictions, p)
        fit!(m, day_games)
    end
    return predictions
end

#Predict with one-ahead for each day for BradleyTerry
function one_ahead!(bt::BradleyTerry, games_original::DataFrame; iterations_players::Int64 = 0, iterations_all::Int64 = 3, prediction_function = predict)
    games = sort(games_original, :Period)
    predictions = Float64[]

    if !in("Handicap", names(games))
        games.Handicap = 0.0
    end

    if !in("Var", names(games))
        games.Var = 0.0
    end

    #Split into days which are then predicted ahead of time
    for day_games in groupby(games, :Period)
        day_games = DataFrame(day_games)
        p = prediction_function.(Ref(bt), day_games[:, 1], day_games[:, 2])
        predictions = vcat(predictions, p)

		#Add the new games and iterate to fit - first on players who played on that day, then on all
		add_games!(bt, day_games)
		#iterate!(bt, iterations_players, vcat(day_games.P1, day_games.P2))
        iterate!(bt, iterations_all)
    end
    return predictions
end

#Predict games one day at a time, then add and rescan
function one_ahead!(whr::WHR, games_original::DataFrame; prediction_function = predict, iterations_players::Int64 = 0, iterations_all::Int64 = 1, show_ll::Bool = false, verbose = true)
    #games = sort(games_original, :Period)
    games = games_original
    predictions = Float64[]

    if !in("Handicap", names(games))
        games.Handicap = 0.0
    end

    if !in("Var", names(games))
        games.Var = 0.0
    end

    #Split into days which are then predicted ahead of time
    for day_games in groupby(games, :Period)
		p = Float64[]
        day_games = DataFrame(day_games)
		#println("Day number ", day_games.Day[1])

		#use chosen prediction function to give predictions for day

		p = prediction_function.(Ref(whr), day_games.P1, day_games.P2, rating_day = day_games.Period[1])

		#Add the predictions to the array of one-ahead
        predictions = vcat(predictions, p)

		add_games!(whr, day_games, dummy_games = true, verbose = verbose)

		if show_ll
			println("Before iterating LL = : ", loglikelihood(whr))
		end

		if iterations_players != 0
			iterate!(whr, unique(vcat(day_games.P1, day_games.P2)), iterations_players)
		end

		if iterations_all != 0
        	iterate!(whr, iterations_all, exclude_non_ford = false, verbose = verbose)
		end

		if show_ll
			println("After iterating LL = : ", loglikelihood(whr))
		end
    end
    return predictions
end

function fit!(algorithms::Array, games_original::DataFrame; masks = missing)
    #Apply fit algorithms and output results
    if ismissing(masks)
        masks = trues(size(games_original)[1], length(algorithms))
    end

    for (place, algo) in enumerate(algorithms)
        #Set mask for appropriate algo and then fit
        masked = games_original[masks[:, place], :]
        add_and_update!(algo, masked)
        
    end
end

function iterate!(algorithms::Array, iterations_on_all)
    #Iterate on the training set only. Ignore unless a WHR/BT algo
    for algo in algorithms
        if typeof(algo) == WHR
            iterate!(algo, iterations_on_all)
        end
    end
end

function one_ahead!(algorithms::Array, games_original::DataFrame; masks = missing, verbose = true, kwargs...)
    #Apply prediction algorithms in array games in dataframe one day ahead of re-fitting and continuing
    #Games must be sorted by date already

    if ismissing(masks)
        masks = trues(size(games_original)[1], length(algorithms))
    end

    #Set up an array to store predictions
    predictions = zeros(Float64, 0, length(algorithms))
    players_check = zeros(Int64, 0, 3)
    result_check = zeros(Float64, 0)

    grouped = groupby(games_original, :Period)
    group_indices = groupindices(grouped)
    last_day = maximum(games_original.Period)

    #Main loop around groups of days
    for (group_number, day_games) in enumerate(grouped)
        day_games = DataFrame(day_games)
        rating_day = day_games.Period[1]

        if verbose
            IJulia.clear_output(true)
            println("\r Evaluating day ", rating_day, " of ", last_day)
        end

        predictions_day = Array{Union{Missing, Float64}}(missing, size(day_games)[1], length(algorithms))

        #Check first which masks are being applied to each algorithm in this day-group
        mask_loop = map(x -> x == group_number, group_indices)
        mask = masks[mask_loop, :]

        #Now, predict results according to each algo
        for (place, algorithm) in enumerate(algorithms)

            if typeof(algorithm) == WHR && haskey(kwargs, :iterations_players)
                #iterate over players that are going to have their games predicted that day (for WHR only)
                players_today = unique(vcat(day_games.P1[mask[:, place]], day_games.P2[mask[:, place]]))
                players_in_dict_today = players_today[haskey.(Ref(algorithm.playerdayratings), players_today)]
                #iterate over players already in player_dict
                iterate!(algorithm, players_in_dict_today, kwargs[:iterations_players])
                if haskey(kwargs, :full_iteration_every) && group_number % kwargs[:full_iteration_every] == 0
                    iterate!(algorithm, 1)
                end
            end

            #Apply mask to rows of dataframes, any masked out return missing values
            predictions_day[:, place] = predict_or_missing.(Ref(algorithm), day_games.P1, day_games.P2, Ref(rating_day), mask[:, place]; kwargs...)

            masked = day_games[mask[:, place], :]
            add_and_update!(algorithm, masked; kwargs...)
        end

        #Store predictions in ongoing array
        predictions = vcat(predictions, predictions_day)

        #Store game results and players in ongoing arrays to check that we've rated the right ones.
        result_day = day_games.P1_wins  ./ (day_games.P1_wins .+ day_games.P2_wins)

        result_check = vcat(result_check, result_day)
        players_check = vcat(players_check, hcat(day_games.P1, day_games.P2, day_games.Period))
    end

    return predictions, result_check, players_check, algorithms
end

function predict_or_missing(algo, P1, P2, rating_day, mask; prediction_function = predict, kwargs...)
    if mask == 0 return missing
    else
        return prediction_function(algo, P1, P2; rating_day = rating_day)
    end
end


function add_and_update!(elo::Elo, day_games::DataFrame; kwargs...)
    #Elo adding new day games and updating

    if !in("Handicap", names(day_games))
        day_games.Handicap = 0.0
    end

    fit!(elo, day_games)
end

function add_and_update!(glicko::Glicko, day_games::DataFrame; kwargs...)
   #Glicko adding new day games and updating
    fit!(glicko, day_games)
end


function add_and_update!(whr::WHR, day_games::DataFrame; kwargs...)

    defaults = (;  iterations_players = 1, iterations_all = 0)
    settings = merge(defaults, kwargs)

    if !in("Handicap", names(day_games))
        day_games.Handicap = 0.0
    end

    if !in("Var", names(day_games))
        day_games.Var = 0.0
    end


    add_games!(whr, day_games, dummy_games = true, verbose = false)

    #iterate over players that have had their games added that day
    players_today = unique(vcat(day_games.P1, day_games.P2))
    iterate!(whr, players_today, settings[:iterations_players])

    #and/or iterate over all players
    iterate!(whr, settings[:iterations_all], exclude_non_ford = false, verbose = false)
end


