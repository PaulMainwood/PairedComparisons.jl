#Predicts with one-ahead for each day for Elo
function one_ahead!(m::Elo, games::DataFrame; prediction_function = predict)
    #Predicts with one-ahead for each day
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

#Predicts with one-ahead for each day for Glicko
function one_ahead!(m::Glicko, games::DataFrame; prediction_function = predict)
    sort!(games, 5)
    predictions = Float64[]
    #Split into days which are then predicted ahead of time
    for day_games in groupby(games, 5)
        day_games = DataFrame(day_games)
        p = prediction_function.(Ref(m), day_games[:, 1], day_games[:, 2], day_games[1, 5])
        predictions = vcat(predictions, p)
        fit!(m, day_games)
    end
    return predictions
end

#Predict with one-ahead for each day for BradleyTerry
function one_ahead!(bt::BradleyTerry, games::DataFrame; iterations_players::Int64 = 0, iterations_all::Int64 = 3, prediction_function = predict)
    sort!(games, 5)
    predictions = Float64[]
    #Split into days which are then predicted ahead of time
    for day_games in groupby(games, 5)
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
function one_ahead!(whr::WHR, games::DataFrame; prediction_function = predict, iterations_players::Int64 = 0, iterations_all::Int64 = 1, show_ll::Bool = false)
    sort!(games, 5)
    predictions = Float64[]

    #Split into days which are then predicted ahead of time
    for day_games in groupby(games, 5)
		p = Float64[]
		day =
        day_games = DataFrame(day_games)
		#println("Day number ", day_games.Day[1])

		#use chosen prediction function to give predictions for day

		p = prediction_function.(Ref(whr), day_games[!, 1], day_games[!, 2]; rating_day = day_games[1, 5])

		#Add the predictions to the array of one-ahead
        predictions = vcat(predictions, p)

		add_games!(whr, day_games, dummy_games = true)

		if show_ll
			println("Before iterating LL = : ", loglikelihood(whr))
		end

		if iterations_players != 0
			iterate!(whr, unique(vcat(day_games.P1, day_games.P2)), iterations_players)
		end

		if iterations_all != 0
        	iterate!(whr, iterations_all, exclude_non_ford = false)
		end

		if show_ll
			println("After iterating LL = : ", loglikelihood(whr))
		end
    end
    return predictions
end
