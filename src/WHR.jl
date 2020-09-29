##############################################################################################################################################
# Implements the Whole History Rating algorithm from Remi Coulom as outlined in the paper: https://www.remi-coulom.fr/WHR/WHR.pdf
#
# The "predict" function is mine, based on my understanding of the algorithm. It gives good results, but I have not checked it with the paper
# and it does not match algorithms given elsewhere (e.g., in the Ruby implementation here: https://github.com/goshrine/whole_history_rating)
#
# This algorithm is also several orders of magnitude faster than the open-source versions (Ruby, Python).
##############################################################################################################################################

import Distributions
import QuadGK
using LinearAlgebra
using IJulia
using LoopVectorization

struct WHR
	playerdayratings::Dict
	playerdaygames::Dict
	default_rating::Float64
	default_score::Float64
	w2::Float64
end

function WHR(;playerdayratings = Dict{Int64, Dict{Int64, Float64}}(), playerdaygames = Dict{Int64, Dict{Int64, Dict{Int64, Tuple{Int64, Int64}}}}(), default_rating = 0.0, default_score = 1.0, w2 = 0.000424)
	return WHR(playerdayratings, playerdaygames, default_rating, default_score, w2)
end

#Add games to an WHR object
function add_games!(whr::WHR, original_games::DataFrame; dummy_games::Bool = true, verbose = false)

    #Duplicate whole of new games rating dataframe with reversed results
	games = dupe_for_rating(original_games)
	P1, P2, P1_wins, P2_wins, Day = games.P1, games.P2, games.P1_wins, games.P2_wins, games.Period

	games_added = 0
	players_added = 0

	for row in 1:length(P1)
		#If this row introduces a new player1, create their (empty) entry in the playerdaygames dictionary, create a new playerday and add games
		if !haskey(whr.playerdaygames, P1[row])
			add_player!(whr::WHR, P1[row])
			players_added += 1
		end

		#If this row introduces a new gameday for a player, add that.
		if !haskey(whr.playerdaygames[P1[row]], Day[row])
			add_gameday!(whr::WHR, P1[row], Day[row])
		end

		#If this row introduces a new opponent in an existing gameday for a player, add that.
		if !haskey(whr.playerdaygames[P1[row]][Day[row]], P2[row])
			add_each_game!(whr::WHR, P1[row], Day[row], P2[row], P1_wins[row], P2_wins[row])
			games_added += 1
		end

		#If none of the conditions hold, then game likely a duplicate and should ignore.

	end
	verbose && println("Added ", games_added ÷ 2, " new games out of ", length(P1) ÷ 2)
	verbose && println("Added ", players_added, " new players.")
end

		

function add_player!(whr::WHR, player::Int64)
	#Create an empty Dictionary for a new player: day => gamesarray and a new ratings Dictionary day => rating
	whr.playerdaygames[player] = Dict{Int64, Dict{Int64, Tuple{Int64, Int64}}}()
	whr.playerdayratings[player] = Dict{Int64, Float64}()
end

function add_gameday!(whr::WHR, player::Int64, day::Int64)
	#Create an empty Dict for a new gameday and a new rating for that gameday
	#Note that one could come up with a better starting point, e.g., the previous rating
	whr.playerdaygames[player][day] = Dict{Int64, Tuple{Int64, Int64}}()
	whr.playerdayratings[player][day] = whr.default_rating
end

function add_each_game!(whr::WHR, P1::Int64, day::Int64, P2::Int64, P1_wins::Int64, P2_wins::Int64)
	#Create a new opponent (P2) and add the scores of the game as a Tuple
	whr.playerdaygames[P1][day][P2] = (P1_wins, P2_wins)
end


function iterate!(whr::WHR, iterations::Int64; exclude_non_ford::Bool = false, delta::Float64 = 0.001, verbose = false, show_ll = false, ll_every = 1)

	#Check and get rid of any players not Ford-connected

	if exclude_non_ford
		non_ford = check_ford(whr)
		connected_players = filter(x -> ∉(first(x), non_ford) , bt.playergames)
		println(length(non_ford), " players excluded due to not being connected. ", length(connected_players), " players being rated.")
	else
		connected_players = whr.playerdaygames
	end

	#Iterate over players selected
	for i = 1:iterations
		for (player, _) in connected_players
			if length(whr.playerdaygames[player]) == 1
				#Need to use a separate update method if we only have one day of games
				update_rating_1dim!(whr, player)
			else
				#If we have several days of games use general update method
				update_rating_ndim!(whr, player, delta = delta)
			end
		end
		verbose && println("Iteration on whole WHR: ", i)


		if show_ll 
			if i % ll_every  == 0
				println("Log likelihood = ", log_likelihood(whr))
			end
		end
	end
end

function iterate!(whr::WHR, players::Array{Int64}, iterations::Int64; exclude_non_ford::Bool = false, delta::Float64 = 0.001)
	#Iterate over players selected
	for i = 1:iterations
		for player in players
			if length(whr.playerdaygames[player]) == 1
				#Need to use a separate update method if we only have one day of games
				update_rating_1dim!(whr, player)
			else
				#If we have several days of games use general update method
				update_rating_ndim!(whr, player, delta = delta)
			end
		end
	end
	#println(iterations, " iterations on added players (", players,") completed.")
end

function iterate_to_convergence!(whr::WHR; step::Int64 = 1, delta::Float64 = 0.001, tolerance::Float64 = 0.00000001)
	ll_percentage = 0.1
	ll = log_likelihood(whr)
	ll_old = ll
	n = 0
	while ll_percentage > tolerance
		n += step
		iterate!(whr, step, delta = delta)
		ll = log_likelihood(whr)
		diff = abs(ll_old) - abs(ll)
		ll_percentage = - (ll - ll_old) / ll_old
		println("iteration: ", n, ". Log-likelihood: ", ll, ". Change: ", ll_percentage)
		ll_old = ll
	end
end


#Use Newton-Raphson method
function update_rating_ndim!(whr::WHR, player::Int64; delta::Float64 = 0.001)

	#Get all the players' days played and initial ratings
	#Note: using "get" might allow to set initial rating at average of others - big speed-up

	playerdays = sort(unique(collect(keys(whr.playerdaygames[player])))) #All the days the player played
	playerratings = [whr.playerdayratings[player][day] for day in playerdays] #Array of the old rating the player had on those days

	#Get the sigma2 vector (one-over it in this case)
	s2inv = sigma2inv(playerdays, whr.w2, player)

	#Obtain the log likelihood derivative and second derivative at one sweep
	lld, ll2d = llderivatives(whr, player, playerdays)
	h = hessian(ll2d, s2inv, delta = delta)
	g = gradient(lld, s2inv, playerratings)

	newratings = playerratings - inv(h) * g

	for (i, day) in enumerate(playerdays)
		whr.playerdayratings[player][day] = newratings[i]
	end
end

#Update using Newton-Raphson method for players with only one game
function update_rating_1dim!(whr::WHR, player::Int64)
	playerdays = sort(unique(collect(keys(whr.playerdaygames[player]))))
	lld, ll2d = llderivatives(whr, player, playerdays)
	dr = lld[1] / ll2d[1]
	whr.playerdayratings[player][playerdays[1]] = whr.playerdayratings[player][playerdays[1]] - dr
end

#Broadcast the element-wise calculation of derivative elements over the playerdays
function llderivatives(whr, player, playerdays)
	days_played = length(playerdays)
	lld = zeros(Float64, days_played)
	ll2d = zeros(Float64, days_played)

	#Add the term for 1 dummy win and 1 dummy loss against opponent of rating 0 on first day
	r1 = exp(whr.playerdayratings[player][first(playerdays)])
	lld[1] = (1 - r1) / (1 + r1)
	ll2d[1] = -2 * r1 / (1 + r1)^2

	#Loop round playerdays and add in loglikelihood derivates to vectors
	for (daynum, day) in enumerate(playerdays)
		llde, ll2de = llderivativeselements(whr, player, day)
		lld[daynum] += llde
		ll2d[daynum] += ll2de
	end
	return lld, ll2d
end

#For each playerday, calculate the llderivates elements (first and second derivatives)
function llderivativeselements(whr, player::Int64, day::Int64)
	lld_tally = 0.0
	ll2d_tally = 0.0
	win_tally = 0.0
	a = exp(get(whr.playerdayratings[player], day, 0.0))

	for (opponent, results) in whr.playerdaygames[player][day]
		b = exp(get(whr.playerdayratings[opponent], day, 0.0))
		win_tally += results[1]
		sumab = 1.0 / (a + b)
		lld_tally_add = (results[1] + results[2]) * sumab

		lld_tally += lld_tally_add
		ll2d_tally += lld_tally_add * b * sumab
	end

	return (win_tally - (a * lld_tally), -a * ll2d_tally)
end

function llelements(whr, player::Int64, day::Int64)
	ll_tally = 0.0
	own_rating = get(whr.playerdayratings[player], day, 0.0)

	for (opponent, result) in whr.playerdaygames[player][day]
		if opponent == 0
			opponent_rating = 1.0
		else
			opponent_rating = get(whr.playerdayratings[opponent], day, 0.0)
		end
		ll_tally += result * own_rating + ((1 - result) * opponent_rating) - log(exp(own_rating) + exp(opponent_rating))
	end

	return ll_tally
end

function log_likelihood(whr::WHR, player)
	#Calculate current log-likelihood for a player history
	player_tally = 0.0
	player_tally = sum(llelements(whr, player, day) for day in keys(whr.playerdaygames[player]))
	return player_tally
end

function log_likelihood(whr::WHR)
	#Calculate current total log-likelihood for all players
	full_tally = sum(log_likelihood(whr::WHR, player) for player in keys(whr.playerdayratings))
	return full_tally
end

function sigma2inv(playerdays, w2::Float64, player::Int64)
	l = length(playerdays) - 1
	s2inv = zeros(Float64, l)
	for i in 1:l
		s2inv[i] = 1 / (w2 * (playerdays[i + 1] - playerdays[i]))
	end
	return s2inv
end

function hessian(ll2d, s2inv; delta::Float64 = 0.001)
	#Construct tridiagonal hessian matrix

	l = length(ll2d)

	prior = zeros(Float64, l)

	prior[1] = s2inv[1]
	for i in 2:l-1
		prior[i] = s2inv[i - 1] + s2inv[i]
	end
	prior[l] = s2inv[l - 1]
	d = ll2d .- prior .- delta
	return Tridiagonal(s2inv, d, s2inv)
end

function gradient(lld, s2inv, playerratings)
	#Construct gradient vector
	l = length(lld)	
	prior = zeros(Float64, l)
	prior[1] = (playerratings[2] - playerratings[1]) * s2inv[1]
	for i in 2:(l-1)
		prior[i] = (playerratings[i + 1] - playerratings[i]) * s2inv[i] - (playerratings[i] - playerratings[i - 1]) * s2inv[i - 1]
	end
	prior[l] = - (playerratings[l] - playerratings[l - 1]) * s2inv[l - 1]

	#prior = vcat(difference ./ s2, 0.0) - vcat(0.0, difference ./ s2)
	return lld .+ prior
end

function check_ford(whr::WHR)
	#Check for players who have either never won or never lost
	unconnected = []
	for (player, games) in bt.playergames
		if length(filter(y -> y[3] == 0.0, games)) == 0
			unconnected = vcat(unconnected, player)
		end
		if length(filter(y -> y[3] == 1.0, games)) == 0
			unconnected = vcat(unconnected, player)
		end
	end
	return unconnected
end

function covariance(whr::WHR, player::Int64)

	#Following Appendix B in the WHR paper -- there's almost certainly a better way to do this in Julia
	playerdays = sort(unique(collect(keys(whr.playerdaygames[player]))))
	playerratings = [whr.playerdayratings[player][day] for day in playerdays]

	s2inv = sigma2inv(playerdays, whr.w2, player)
	lld, ll2d = llderivatives(whr, player, playerdays)
	h = hessian(ll2d, s2inv)
	g = gradient(lld, s2inv, playerratings)

	n = length(playerdays)

	a = zeros(Float64, n)
	d = zeros(Float64, n)
	b = zeros(Float64, n)

	d[1] = h[1, 1]
	if size(h)[1] > 2
		b[1] = h[1, 2]
	else
		b[1] = 0.0
	end

	for i in range(2, stop = n)
		a[i] = h[i, i - 1] / d[i - 1]
		d[i] = h[i, i] - a[i] * b[i - 1]
		if i < n
			b[i] = h[i, i + 1]
		end
	end

	dp = zeros(Float64, n)
	dp[n] = h[n, n]
	bp = zeros(Float64, n)
	bp[n] = h[n, n - 1]

	ap = zeros(Float64, n)
	for i in range(n - 1, step = -1, stop = 1)
		ap[i] = h[i, i + 1] / dp[i + 1]
		dp[i] = h[i, i] - ap[i] * bp[i + 1]
		if i > 1
			bp[i] = h[i, i-1]
		end
	end

	v = zeros(Float64, n)
	for i in range(1, stop = n - 1)
		v[i] = dp[i + 1]/(b[i] * bp[i + 1] - d[i] * dp[i + 1])
	end
	v[n] = -1 / d[n]

	ev = - a[2:n] .* v[1:n - 1]

	return Bidiagonal(v, ev, :U)
end

function uncertainty(whr::WHR, player::Int64)
	#Create a dictionary of uncertainties around each rating for a player
	if haskey(whr.playerdaygames, player)
		playerdays = sort(unique(collect(keys(get(whr.playerdaygames, player, [0])))))
		ndays = length(whr.playerdaygames[player])
	else
		playerdays = [0]
		ndays = 0
	end

	if ndays > 1
		c = covariance(whr, player)
		u = diag(c)
	else
		u = [1.0]
	end

	d = Dict(playerdays .=> u)
	return d
end

function ratings(whr::WHR, player::Int64)
	#Return the two ratings dictionaries - of ratings, and of the uncertainties around them
	return get(whr.playerdayratings, player, whr.default_rating), uncertainty(whr, player)
end

function rating(whr::WHR, player::Int64; rating_day::Int64)
	#Return the player's rating on a day, and the uncertainties it has, projected forward from last rating period
	
	if haskey(whr.playerdayratings, player)
		rating_dict, var_dict = ratings(whr, player)
	else
		return whr.default_rating, 1.0
	end


	if haskey(rating_dict, rating_day)
		rating = rating_dict[rating_day]
		var = var_dict[rating_day]
	else
		b = collect(keys(rating_dict))
		if isempty(b[b .< rating_day])
			rating = whr.default_rating
			var = 1.0
		else
			last_day_rated = maximum(b[map(x -> x < rating_day, b)])
			rating = rating_dict[last_day_rated]
			var = get(var_dict, last_day_rated, 0)  + abs(last_day_rated - rating_day) * abs(whr.w2)
		end
	end
	return rating, var
end

function rating(whr::WHR, P1::Int64, P2::Int64; rating_day::Int64 = 0)
	#Return a pair of players' ratings as a pair of tuples
	return rating(whr, P1; rating_day = rating_day), rating(whr, P2; rating_day = rating_day)
end

function predict(whr::WHR, P1::Int, P2::Int; rating_day = missing)
	#Predict with logitnormal distribution, pulling forward the variance to the rating day (default, present day)

	#If no rating_day provided, use the last day on which either player was rated
	if ismissing(rating_day)
		rating_day = maximum((maximum(keys(whr.playerdaygames[P1])), maximum(keys(whr.playerdaygames[P2])), 0))
	end

	#If no games, set default r value
	if !haskey(whr.playerdaygames, P1) || length(whr.playerdaygames[P1]) == 0
		r1 = 0.0
		var1 = 1.0
	else
		lastdayP1 = maximum(keys(whr.playerdaygames[P1]))
		r1, var1_lastday = rating(whr, P1, lastdayP1)
		var1 = abs(var1_lastday) + abs(rating_day - lastdayP1) * abs(whr.w2)
	end

	if !haskey(whr.playerdaygames, P2) || length(whr.playerdaygames[P2]) == 0
		r2 = 0.0
		var2 = 1.0
	else
		lastdayP2 = maximum(keys(whr.playerdaygames[P2]))
		r2, var2_lastday = rating(whr, P2, lastdayP2)
		var2 = abs(var2_lastday) + abs(rating_day - lastdayP2) * abs(whr.w2)
	end

	#Difference between two normally distributed RVs is normally distributed with mean of the difference of the two means, and variance as sum of the variances.
	#For the probability, we are looking for the logistic function of this normal distribution: given by the mean of the logit-normal (I hope).
	#Using approximation here with series of 10 terms
	return mean_logitnormal_approx(r1 - r2, var1 + var2, 10)
end

function mean_logitnormal(mu, var)
	#Returns the mean of the logitnormal. No analytic way, so using Gaussian quadrature
	#Can be made much faster with approximations, but not a performance-critical piece (I hope).
	dist = Distributions.LogitNormal(mu, sqrt(var))
	prob_win = QuadGK.quadgk(x -> x * Distributions.pdf(dist, x), 0, 1)[1]
    return prob_win
end

function mean_logitnormal_approx(mu, var, K)
	#Approximate function for logitnormals - about 1000 times faster and also more stable than numerical quadrature
    cumulative = 0.0
    d = Distributions.Normal(mu, var)
    for i in 1:K-1
        cumulative += std_logistic(Distributions.quantile(d, i/K))
    end
    return cumulative / (K-1)
end

function std_logistic(x)
    return 1 / (1 + exp(-x))
end