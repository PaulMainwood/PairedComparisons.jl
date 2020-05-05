
function exp_smoothing(whr::WHR, player::Int64, projection_day::Int64; α::Float64 = 0.0, β::Float64 = 0.0)
	#Predict a rating "plus" days after the last game played using single exponential smoothing
    sorted = sort(whr.playerdayratings[player])

	if β == 0.0
		return single_exp_smoothing_prediction(sorted, α, projection_day)
	else
		return double_exp_smoothing_prediction(sorted, α, β, projection_day)
	end
end

function single_exp_smoothing_prediction(sorted, α, projection_day; whole_dict::Bool = false)
	#Project on single exponential smoothing
	y = 0.0
	k = collect(keys(sorted))
	pred = Dict(first(k) => sorted[first(k)])
    for day in first(k):projection_day - 1

		if haskey(sorted, day)
            y = sorted[day]
        end

        pred[day + 1] = α * y + (1 - α) * pred[day]

    end

	if whole_dict
		return pred
	else
		return pred[projection_day]
	end
end

function double_exp_smoothing_prediction(sorted, α, γ, projection_day; whole_dict::Bool = false)
	# Algorithm for improved double exponential smoothing prediction from Hanzak's paper
	k = collect(keys(sorted))
	firstday = first(k)

	#Initialise dictionaries with initial values
	S_dict = Dict(firstday => sorted[firstday])
	T_dict = Dict(firstday => 0.0)

	y_hat = Dict(firstday => sorted[firstday])

	q = (last(k) - first(k)) / length(k)
	alpha_dict = Dict(firstday => 1.0 - (1.0 - α)^q)
	gamma_dict = Dict(firstday => 1.0 - (1.0 - γ)^q)

	if length(k) < 2
	#If only one game, just return current level
		return S_dict[firstday]
	end

	prev_day = firstday
	prev_prev_day = firstday

	for day in (first(k) + 1):projection_day

		#Make prediction for each day before updating with new information

		y_hat[day] = get(S_dict, prev_day, S_dict[firstday]) + (day - prev_day) * get(T_dict, prev_day, T_dict[firstday])

		#And now - if we have a read - update the smoother coefficients and set prev_day to day
		if haskey(sorted, day)
			alpha_dict[day] = update_alpha(alpha_dict, α, day, prev_day)
			gamma_dict[day] = update_gamma(gamma_dict, γ, day, prev_day, prev_prev_day)

			S_dict[day] = S_d(sorted, alpha_dict, S_dict, T_dict, day, prev_day)
			T_dict[day] = T_d(sorted, gamma_dict, S_dict, T_dict, day, prev_day)
			prev_prev_day = prev_day
			prev_day = day
		end
	end

	if whole_dict
		return y_hat
	else
		return y_hat[projection_day]
	end
end

function findpreviousday(day, sorted)
	key = collect(keys(sorted))
	return key[findlast(x -> day - x >= 0, key)]
end

function S_d(sorted, alpha_dict, S_dict, T_dict, day, prev_day)
	alpha = alpha_dict[day]
	return (1 - alpha) * (S_dict[prev_day] + ((day - prev_day) * T_dict[prev_day])) + alpha * sorted[day]
end

function T_d(sorted, gamma_dict, S_dict, T_dict, day, prev_day)
	gamma = gamma_dict[day]
	return (1 - gamma) * T_dict[prev_day] + gamma * (S_dict[day] - S_dict[prev_day]) / (day - prev_day)
end

function update_alpha(alpha_dict, α, day, prev_day)
	return alpha_dict[prev_day] / (alpha_dict[prev_day] + (1 - α)^(day - prev_day))
end

function update_gamma(gamma_dict, γ, day, prev_day, prev_prev_day)
	return gamma_dict[prev_day] / (gamma_dict[prev_day] + (prev_day - prev_prev_day) / (day - prev_day) * (1 - γ)^(day - prev_day))
end

function order_by_recency(whr::WHR)
    maxdays = Dict(player => maximum(keys(days)) for (player, days) in whr.playerdaygames)
    return [i[1] for i in sort(collect(maxdays), by = x -> x[2], rev = true)]
end


function predict_no_uncertainty(whr::WHR, P1::Int64, P2::Int64; rating_day::Int64 = Dates.value(Dates.today() - earliest_day), α::Float64 = 0.0)
	#Generate standard Bradley Terry probability for P1 winning a game
	#NOTE this doesn't allow for drift between last game and now

	if !haskey(whr.playerdaygames, P1)
		return missing
	end
	if !haskey(whr.playerdaygames, P2)
		return missing
	end
	#Set rating to latest rating achieved
	if length(whr.playerdaygames[P1]) == 0
		gammaP1 = 1.0
	else
		dayP1 = maximum(keys(whr.playerdaygames[P1]))

		if α == 0
			gammaP1 = exp(whr.playerdayratings[P1][dayP1])
		else
			gammaP1 = exp(exp_smoothing(whr, P1, rating_day, α = α))
		end
	end

	if length(whr.playerdaygames[P2]) == 0
		gammaP2 = 1.0
	else
		dayP2 = maximum(keys(whr.playerdaygames[P2]))
		if α == 0
			gammaP2 = exp(whr.playerdayratings[P2][dayP2])
		else
			gammaP2 = exp(exp_smoothing(whr, P2, rating_day, α = α))
		end
	end

	last_day_prediction = 1.0 / (1.0 + (gammaP2 / gammaP1))

	return last_day_prediction

end

function predict_with_uncertainty(whr::WHR, P1::Int64, P2::Int64, sets::Int64; rating_day::Int64 = Dates.value(Dates.today() - earliest_day), α::Float64 = 0.0, β::Float64 = 0.0)
#Predict with logitnormal distribution
	P1 = convert(Int64, P1)
	P2 = convert(Int64, P2)

	if !haskey(whr.playerdaygames, P1)
		return missing
	end
	if !haskey(whr.playerdaygames, P2)
		return missing
	end
	#If no games, set default r value
	if length(whr.playerdaygames[P1]) == 0
		r1 = 0.0
		var1 = 1.0
	else
		lastdayP1 = maximum(keys(whr.playerdaygames[P1]))
		if α == 0.0
			r1, var1_lastday = rating(whr, P1, lastdayP1)
			var1 = var1_lastday + (abs(rating_day - lastdayP1) * whr.w2)
		else
			r1 = exp_smoothing(whr, P1, rating_day, α = α, β = β)
			var1 = rating(whr, P1, lastdayP1)[2] + (abs(rating_day - lastdayP1) * whr.w2)
		end
	end

	if length(whr.playerdaygames[P2]) == 0
		r2 = 0.0
		var2 = 1.0
	else
		lastdayP2 = maximum(keys(whr.playerdaygames[P2]))
		if α == 0.0
			r2, var2_lastday = rating(whr, P2, lastdayP2)
			var2 = var2_lastday + (abs(rating_day - lastdayP2) * whr.w2)
		else
			r2 = exp_smoothing(whr, P2, rating_day, α = α, β = β)
			var2 = rating(whr, P2, lastdayP2)[2] + (abs(rating_day - lastdayP2) * whr.w2)
		end
	end
	return predict_logitnormal(r1 - r2, var1 + var2, sets)
end

function predict_with_uncertainty(whr_surfaces::Tuple, weights::Tuple, P1::Int64, P2::Int64, surface::Int64, sets::Int64; rating_day::Int64 = Dates.value(Dates.today() - earliest_day), α::Float64 = 0.0, β::Float64 = 0.0)
	#Predict with logitnormal distribution
	P1 = convert(Int64, P1)
	P2 = convert(Int64, P2)

	combined_prob = predict_with_uncertainty.(Ref(whr_surfaces[4]), P1, P2, sets; rating_day = rating_day, α = α) * (1.0 - weights[surface]) + predict_with_uncertainty.(Ref(whr_surfaces[surface]), P1, P2, sets; rating_day = rating_day, α = α) * weights[surface]
	return combined_prob
end

function predict_logitnormal(mu, var, sets)
	dist = LogitNormal(mu, sqrt(var))
	if sets >= 4
		prob_win, err = quadgk(x -> (x ^ 3) * (1 + (3 * (1 - x)) + (6 * (1 - x)^2)) * pdf(dist, x), 0, 1)
    else
        prob_win, err = quadgk(x -> (3 * x ^ 2 -  2 * x ^ 3) * pdf(dist, x), 0, 1)
    end

    return prob_win
end


function one_ahead_sorted!(whr::WHR, games::DataFrame, iterations_all::Int64; include_uncertainty::Bool = true, alpha::Float64 = 0.0, beta = 0.0, show_ll::Bool = false, w2_variable = (0.0, 0.0, 0.0, Dict()))
    sort!(games, :Day)
    predictions = Float64[]
	count_played_P1 = Int64[]
	count_played_P2 = Int64[]
    #Split into days which are then predicted ahead of time
    for day_games in groupby(games, :Day)
		p = Float64[]
        day_games = DataFrame(day_games)
		#println("Day number ", day_games.Day[1])

		#two different prediction approaches, including uncertainty estimates or not
		if include_uncertainty
			p = predict_with_uncertainty.(Ref(whr), day_games[:, :P1], day_games[:, :P2], day_games[:, :Total_sets], rating_day = day_games.Day[1], α = alpha, β = beta)
		else
			p = predict_no_uncertainty.(Ref(whr), day_games[:, :P1], day_games[:, :P2], rating_day = day_games.Day[1], α = alpha)
		end

		#Add the predictions to the array of one-ahead
        predictions = vcat(predictions, p)

		countP1 = [length(get(whr.playerdaygames, P1, [])) for P1 in day_games[:, :P1]]
		countP2 = [length(get(whr.playerdaygames, P2, [])) for P2 in day_games[:, :P2]]

		count_played_P1 = vcat(count_played_P1, countP1)
		count_played_P2 = vcat(count_played_P2, countP2)

		#Iterate to fit - first on players who played on that day, then on all
		add_games!(whr, day_games, dummy_games = true)

		if show_ll
			println("Before iterating LL = : ", loglikelihood(whr))
		end

		ordered_players = order_by_recency(whr)
        iterate!(whr, ordered_players, iterations_all, exclude_non_ford = false, w2_variable = (0.0, 0.0, 0.0, Dict()))

		if show_ll
			println("After iterating LL = : ", loglikelihood(whr))
		end
    end
    return predictions, count_played_P1, count_played_P2
end

function one_ahead_predict_all!(whr::WHR, games::DataFrame, iterations_all::Int64; alpha::Float64 = 0.0, beta::Float64 = 0.0)
    #Set up arrays to record all prediction types
    predictions_no_uncertainty = Float64[]
	predictions_with_uncertainty = Float64[]
	predictions_exp_smoothing = Float64[]
	predictions_double_exp_smoothing = Float64[]

	#Also counts of previously played games`
	count_played_P1 = Int64[]
	count_played_P2 = Int64[]
    #Split into days which are then predicted ahead of time
    for day_games in groupby(games, :Day)
        day_games = DataFrame(day_games)
		println("Day number ", day_games.Day[1])

		#two different prediction approaches, including uncertainty estimates or not
		p_no_uncertainty = predict_no_uncertainty.(Ref(whr), day_games[:, :P1], day_games[:, :P2], rating_day = day_games.Day[1])

		p_with_uncertainty = predict_with_uncertainty.(Ref(whr), day_games[:, :P1], day_games[:, :P2], day_games[:, :Total_sets], rating_day = day_games.Day[1])

		p_exp_smoothing = predict_with_uncertainty.(Ref(whr), day_games[:, :P1], day_games[:, :P2], day_games[:, :Total_sets], rating_day = day_games.Day[1], α = alpha)

		p_double_exp_smoothing = predict_with_uncertainty.(Ref(whr), day_games[:, :P1], day_games[:, :P2], day_games[:, :Total_sets], rating_day = day_games.Day[1], α = alpha, β = beta)

		#Add the predictions to the array of one-ahead
		predictions_no_uncertainty = vcat(predictions_no_uncertainty, p_no_uncertainty)
		predictions_with_uncertainty = vcat(predictions_with_uncertainty, p_with_uncertainty)
		predictions_exp_smoothing = vcat(predictions_exp_smoothing, p_exp_smoothing)
		predictions_double_exp_smoothing = vcat(predictions_double_exp_smoothing, p_double_exp_smoothing)

		countP1 = [length(get(whr.playerdaygames, P1, [])) for P1 in day_games[:, :P1]]
		countP2 = [length(get(whr.playerdaygames, P2, [])) for P2 in day_games[:, :P2]]

		count_played_P1 = vcat(count_played_P1, countP1)
		count_played_P2 = vcat(count_played_P2, countP2)

		#Iterate to fit - first on players who played on that day, then on all
		add_games!(whr, day_games)
		#iterate!(bt, iterations_players, vcat(day_games.P1, day_games.P2))
        iterate!(whr, iterations_all, exclude_non_ford = false)
    end
    return predictions_no_uncertainty, predictions_with_uncertainty, predictions_exp_smoothing, predictions_double_exp_smoothing, count_played_P1, count_played_P2
end
