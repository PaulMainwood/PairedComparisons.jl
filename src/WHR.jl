import Distributions
import QuadGK
using LinearAlgebra

struct WHR
	playerdayratings::Dict
	playerdaygames::Dict
	default_rating::Float64
	w2::Float64
end

function WHR(;playerdayratings = Dict{Int64, Dict{Int64, Float64}}(), playerdaygames = Dict{Int64, Dict{Int64, Array{Tuple{Int64, Float64}}}}(), default_rating = 0.0, w2 = 0.000454)
	return WHR(playerdayratings, playerdaygames, default_rating, w2)
end

#Add games to an WHR object
function add_games!(whr::WHR, original_games::DataFrame; dummy_games::Bool=true)

	games = original_games[!, (1:5)]
    #Duplicate whole rating dataframe with reversed results
    games = dupe_for_rating(games)

	P1 = games[!, 1]
	P2 = games[!, 2]
	P1_wins = games[!, 3]
	P2_wins = games[!, 4]
	Day = games[!, 5]

	games_added = 0
	players_added = 0
	players_to_iterate = Int64[]

	for row in 1:length(P1)
		#If this row introduces a new player, create their (empty) entry in the playerdaygames dictionary, create a new playerday and add dummy games
		if !haskey(whr.playerdaygames, P1[row])

			add_player!(whr::WHR, P1[row])
			add_gameday!(whr::WHR, P1[row], Day[row])
			whr.playerdaygames[P1[row]][Day[row]] = vcat([(P2[row], 1.0) for i in 1:P1_wins[row]], [(P2[row], 0.0) for j in 1:P2_wins[row]])
			games_added += 1
			#Add dummy games if toggle is on
			if dummy_games
				add_dummy_games_whr!(whr, P1[row])
			end

			players_added += 1
		end

		#If this row just introduces a new gameday, create their dictionary for this gameday
		if !haskey(whr.playerdaygames[P1[row]], Day[row])
			add_gameday!(whr::WHR, P1[row], Day[row])
		end

		#Now add the game to the appropriate gameday for the appropriate player
		new_games = vcat([(P2[row], 1.0) for i in 1:P1_wins[row]], [(P2[row], 0.0) for j in 1:P2_wins[row]])
		destination = whr.playerdaygames[P1[row]][Day[row]]

		#Check games are not already there, and if not add them and count them
		if !in(first(new_games), destination)
			whr.playerdaygames[P1[row]][Day[row]] = vcat(destination, new_games)
			games_added += 1
			#players_to_iterate = vcat([P1[row], P2[row]], players_to_iterate)
		else
			nothing
		end
	end
	println("Added ", games_added, " new games out of ", length(P1))
	println("Added ", players_added, " new players.")
	#return unique(players_to_iterate)
end

function add_player!(whr::WHR, player::Int)
	#Create an empty Dictionary for a new player: day => gamesarray and a new ratings Dictionary day => rating
	whr.playerdaygames[player] = Dict{Int, Array{Tuple{Int, Float64}}}()
	whr.playerdayratings[player] = Dict{Int, Float64}()
end

function add_gameday!(whr::WHR, player::Int, day::Int)
	#Create an empty Array for a new gameday: Array((P2, Result)) and a new rating for that gameday
	#Note that one could come up with a better starting point, e.g., the last rating
	whr.playerdaygames[player][day] = Array{Tuple{Int, Float64}}[]
	whr.playerdayratings[player][day] = whr.default_rating
end

function add_dummy_games_whr!(whr::WHR, i::Int)
	firstday = minimum(keys(whr.playerdaygames[i]))
	if whr.playerdaygames[i][firstday][1] == 0
		#if dummy games exist as the earliest games, do nothing
		nothing
	else
		#if dummy games are not the first games, then add them
		whr.playerdaygames[i][firstday] = vcat([(0, 1.0), (0, 0.0)], whr.playerdaygames[i][firstday])
		#if we have dummy games but they are not the first ones then Delete existing dummy games and create new ones
		#ADD THIS. DOES NOT MATTER IF ALWAYS ADDING FUTURE GAMES BUT WILL MATTER LATER
	end
end

function iterate!(whr::WHR, iterations::Int64; exclude_non_ford::Bool = false, delta::Float64 = 0.001)

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
		#println(loglikelihood(bt, connected_players))
		end
		println("Iteration on whole WHR: ", i)
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
	ll = loglikelihood(whr)
	ll_old = ll
	n = 0
	while ll_percentage > tolerance
		n += step
		iterate!(whr, step, delta = delta)
		ll = loglikelihood(whr)
		diff = abs(ll_old) - abs(ll)
		ll_percentage = - (ll - ll_old) / ll_old
		println("iteration: ", n, ". Log-likelihood: ", ll, ". Change: ", ll_percentage)
		ll_old = ll
	end
end


#Use Newton-Raphson method
function update_rating_ndim!(whr::WHR, player::Int64; delta::Float64 = 0.001)

	#Get all the players' days played and initial ratings
	#Note: using "get" might allow to set initial rating at average of others - speed-up
	playerdays = sort(unique(collect(keys(whr.playerdaygames[player]))))
	playerratings = [whr.playerdayratings[player][day] for day in playerdays]

	#Get the sigma2 vector
	s2 = sigma2(playerdays, whr.w2, player)

	#Obtain the log likelihood derivative and second derivative at one sweep
	lld, ll2d = llderivatives(whr, player, playerdays)
	h = hessian(ll2d, s2, delta = delta)
	g = gradient(lld, s2, playerratings)

	newratings = playerratings - inv(h) * g

	for (i, day) in enumerate(playerdays)
		whr.playerdayratings[player][day] = newratings[i]
	end
end

#Update using Newton-Raphson method for just one player
function update_rating_1dim!(whr::WHR, player::Int64)
	playerdays = sort(unique(collect(keys(whr.playerdaygames[player]))))
	lld, ll2d = llderivatives(whr, player, playerdays)
	dr = lld[1] / ll2d[1]
	whr.playerdayratings[player][playerdays[1]] = whr.playerdayratings[player][playerdays[1]] - dr
end

#Broadcast the element-wise calculation of derivative elements over the playerdays
function llderivatives(whr, player, playerdays)
	days_played = length(playerdays)
	lld = Array{Float64, 1}(undef, days_played)
	ll2d = Array{Float64, 1}(undef, days_played)
	#Loop round playerdays and add in loglikelihood derivates to vectors
	for (daynum, day) in enumerate(playerdays)
		llde, ll2de = llderivativeselements(whr, player, day)
		lld[daynum] = llde
		ll2d[daynum] = ll2de
	end
	return lld, ll2d
end

#For each playerday, calculate the llderivates elements (first and second derivatives)
function llderivativeselements(whr, player::Int64, day::Int64)
	lld_tally = 0.0
	ll2d_tally = 0.0
	win_tally = 0.0
	a = exp(get(whr.playerdayratings[player], day, 0.0))

	for (opponent, result) in whr.playerdaygames[player][day]
		if opponent == 0
			b = 1.0
		else
			b = exp(get(whr.playerdayratings[opponent], day, 0.0))
		end
		win_tally += result
		lld_tally += 1.0 / (a + b)
		ll2d_tally += b / (a + b)^2
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

function loglikelihood(whr::WHR, player)
	#Calculate current log-likelihood for a player history
	player_tally = 0.0
	player_tally = sum(llelements(whr, player, day) for day in keys(whr.playerdaygames[player]))
	return player_tally
end

function loglikelihood(whr::WHR)
	#Calculate current total log-likelihood for all players
	full_tally = sum(loglikelihood(whr::WHR, player) for player in keys(whr.playerdayratings))
	return full_tally
end


function sigma2(playerdays, w2::Float64, player::Int64)
	#Vector of n-1 expressions for drift from the Weiner process between ndays in which the player plays games
	return [w2 * (playerdays[i + 1] - playerdays[i]) for i in 1:(length(playerdays) - 1)]
end

function sigma2(playerdays, a::Float64, b::Float64, c::Float64, d::Float64, playerdobs::Dict, player::Int64)
	#Sigma2 with declining w2 factor as player becomes more experienced
	age_in_days = playerdays .- get(player_dobs, player, -3650)
	return [a * (1.0 - b / (1.0 + exp(- c * (age_in_days[i + 1] - d)))) * (playerdays[i + 1] - playerdays[i]) for i in 1:(length(playerdays) - 1)]
end

function hessian(ll2d, s2; delta::Float64 = 0.001)
	#Construct tridiagonal hessian matrix
	invs2 = 1.0 ./ s2
	prior = -1 .* (vcat(0.0, invs2) .+ vcat(invs2, 0.0))
	d = ll2d .+ prior .- delta
	dl = invs2
	du = invs2
	return Tridiagonal(dl, d, du)
end

function gradient(lld, s2, playerratings)
	#Construct gradient vector
	difference = diff(playerratings)
	prior = vcat(difference ./ s2, 0.0) - vcat(0.0, difference ./ s2)
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

	s2 = sigma2(playerdays, whr.w2, player)
	lld, ll2d = llderivatives(whr, player, playerdays)
	h = hessian(ll2d, s2)
	g = gradient(lld, s2, playerratings)

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

	ev = -a .* v
	return Bidiagonal(v, ev, :U)
end

function uncertainty(whr::WHR, player::Int64)
	#Create a dictionary of uncertainties around each rating for a player
	playerdays = sort(unique(collect(keys(whr.playerdaygames[player]))))
	ndays = length(whr.playerdaygames[player])

	if ndays > 1
		c = covariance(whr, player)
		u = diag(c)
	else
		u = [1.0]
	end

	d = Dict(playerdays .=> u)
	return d
end

function rating(whr::WHR, player::Int64)
	#Return the two ratings dictionaries - of ratings, and of the uncertainties around them
	return whr.playerdayratings[player], uncertainty(whr, player)
end

function rating(whr::WHR, player::Int64, day::Int64)
	#Return the player's rating on a day, and the uncertainties it has
	return whr.playerdayratings[player][day], uncertainty(whr, player)[day]
end

function predict(whr::WHR, P1::Int64, P2::Int64; rating_day = missing)
	#Predict with logitnormal distribution, pulling forward the variance to the rating day (default, present day)

	if !haskey(whr.playerdaygames, P1)
		return missing
	end
	if !haskey(whr.playerdaygames, P2)
		return missing
	end

	#If no rating_day provided, use the last day on which either player was rated
	if ismissing(rating_day)
		rating_day = maximum((maximum(keys(whr.playerdaygames[P1])), maximum(keys(whr.playerdaygames[P2])), 0))
	end

	#If no games, set default r value
	if length(whr.playerdaygames[P1]) == 0
		r1 = 0.0
		var1 = 1.0
	else
		lastdayP1 = maximum(keys(whr.playerdaygames[P1]))
		r1, var1_lastday = rating(whr, P1, lastdayP1)
		var1 = var1_lastday + (abs(rating_day - lastdayP1) * whr.w2)
	end

	if length(whr.playerdaygames[P2]) == 0
		r2 = 0.0
		var2 = 1.0
	else
		lastdayP2 = maximum(keys(whr.playerdaygames[P2]))
		r2, var2_lastday = rating(whr, P2, lastdayP2)
		var2 = var2_lastday + (abs(rating_day - lastdayP2) * whr.w2)
	end
	#Difference between two normally distributed RVs is normally distributed with mean of the difference of the two means, and variance as sum of the variances.
	#For the probability we are looking for the logistic function of this normal distribution: given by the mean of the logit-normal (I hope).
	return mean_logitnormal(r1 - r2, var1 + var2)
end

function mean_logitnormal(mu, var)
	#Returns the mean of the logitnormal. No analytic way, so using Gaussian
	#quadrature. Can be made much faster with approximations, but not a performance-critical piece (I hope).
	dist = Distributions.LogitNormal(mu, sqrt(var))
	prob_win = QuadGK.quadgk(x -> x * Distributions.pdf(dist, x), 0, 1)[1]
    return prob_win
end
