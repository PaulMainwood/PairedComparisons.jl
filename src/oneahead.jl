#Predicts with one-ahead for each day for Elo
function one_ahead!(m::Elo, games_original::DataFrame; prediction_function = predict)
    #Predicts with one-ahead for each day
    games = sort(games_original, :Period)
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
    sort!(games, :Period)
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
		p = Float64[]
        day_games = DataFrame(day_games)
		#println("Day number ", day_games.Day[1])

		#use chosen prediction function to give predictions for day

		p = prediction_function.(Ref(whr), day_games.P1, day_games.P2, day_games.Var, rating_day = day_games.Period[1])

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
