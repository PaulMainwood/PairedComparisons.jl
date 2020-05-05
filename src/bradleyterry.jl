
struct BradleyTerry
	rating::Dict
	playergames::Dict
	default_rating::Float64
end

function BT(; rating = Dict{Int64, Float64}(), playergames = Dict(), default_rating = 1.0)
    #Set up a new Elo struct, either using key words or defaults if not provided
    return BradleyTerry(rating, playergames, default_rating)
end

#Add games to a BradleyTerry object
function add_games!(bt::BradleyTerry, games::DataFrame; dummy_games::Bool = true)
    #Duplicate whole rating dataframe with reversed results
    games = dupe_for_rating(games)

    #Split up by player (P1) and create Dictionary of games belonging to each player (won and lost)
    for player in groupby(games, 1)

		#Add all games for that player
		p = Tuple.(eachrow(player))
		i = player[1,1]
        bt.playergames[i] = vcat(get(bt.playergames, i, []), p)

		sort!(bt.playergames[i], by = x -> x[4])

		#See whether we've got our dummy games in right place
		if dummy_games
			bt.playergames[i] = add_dummy_games(i, bt.playergames[i])
		end
    end
end

function add_dummy_games(i, games)
	if games[1][2] == 0
		#if dummy games exist as the earliest games, do nothing
		return games
	else
		#if dummy games are not the first games, then add them
		firstday = games[1][4]
		return vcat([(i, 0, 1.0, firstday, 1), (i, 0, 0.0, firstday, 1)], games)
		#if we have dummy games but they are not the first ones then Delete existing dummy games and create new ones
		#ADD THIS. DOES NOT MATTER IF ALWAYS ADDING FUTURE GAMES BUT WILL MATTER LATER
	end
end

#Iterate the ratings for all players
function iterate!(bt::BradleyTerry, iterations::Int64; exclude_non_ford::Bool = false)

	#Check and get rid of any players not Ford-connected
	if exclude_non_ford
		non_ford = check_ford(bt)
		connected_players = filter(x -> ∉(first(x), non_ford) , bt.playergames)
		#println(length(non_ford), " players excluded due to not being connected. ", length(connected_players), " players being rated.")
	else
		connected_players = bt.playergames
	end

	#println("Initial loglikelihood ", loglikelihood(bt, connected_players))

	#Iterate over players selected
	for i = 1:iterations
		for (player, _) in connected_players
			bt.rating[player] = update_rating(bt, player)
		end
		#println(loglikelihood(bt, connected_players))
	end
end

#Iterate the rating of selection of players
function iterate!(bt::BradleyTerry, iterations::Int64, players)
	for i = 1:iterations
		for player in players
			bt.rating[player] = update_rating(bt, player)
		end
		println(loglikelihood(bt, players))
	end
end

#Scale average of ratings back to 1.0
function normalise!(ratingsdict)
	average_rating = mean(rating for (player, rating) in ratingsdict)
	println(average_rating)
	for (k, v) in ratingsdict
    	ratingsdict[k] = v / average_rating
	end
end

#Update function allowing for period in which there may be duplicate games
function update_rating(bt::BradleyTerry, player::Int64)
	p = sum(w for (_, _, w, _) in bt.playergames[player])
	opponents = unique(j for (_, j, _, _) in bt.playergames[player])
	q = sum(zermelement(bt, player, opponent) for opponent in opponents)
	return p / q
end

function loglikelihood(bt::BradleyTerry; players = bt.playergames)
	tally = 0.0
	for (player, games) in players
		wonplayergames = filter(y -> y[3] == 1.0, games)
		tally += sum(llelement(bt, player, opponent) for (_, opponent, _, _) in wonplayergames)
		tally
	end
	return tally
end

function llelement(bt::BradleyTerry, i::Int64, j::Int64)
	j == 0 && return 1.0 * (log(get(bt.rating, i, bt.default_rating)) - log(get(bt.rating, j, bt.default_rating) + 1))
	player_won_games = length(filter(y -> y[2] == j && y[3] == 1.0, get(bt.playergames, i, [])))
	return player_won_games * (log(get(bt.rating, i, bt.default_rating)) - log(get(bt.rating, i, bt.default_rating) + get(bt.rating, j, bt.default_rating)))
end

#Calculate the element of the Zermelo algorithm given in Hunter paper
function zermelement(bt::BradleyTerry, i::Int64, j::Int64)
	i == j && return 0.0 #Don't include self
	#j == 0 && return 2 / (bt.rating[i] + 1.0) #Give quick answer for dummy games where all players won and lost to player 0 with rating 1.
	Nij = length(filter(y -> y[2] == j, get(bt.playergames, i, [])))  #Number of times players have played one another
	Nij == 0 && return 0.0 #If no matches, immediately return zero
	return Nij / (get(bt.rating, i, bt.default_rating) + get(bt.rating, j, bt.default_rating))
end

#Predict games one day at a time, then add and rescan
function one_ahead!(bt::BradleyTerry, games::DataFrame, iterations_players::Int64, iterations_all::Int64)
    sort!(games, :Day)
    predictions = Float64[]
    #Split into days which are then predicted ahead of time
    for day_games in groupby(games, :Day)
        day_games = DataFrame(day_games)
		println("Day number ", day_games[1, 4])
        p = predict.(Ref(bt), day_games[:, :P1], day_games[:, :P2])
        predictions = vcat(predictions, p)

		#Add the new games and iterate to fit - first on players who played on that day, then on all
		add_games!(bt, day_games)
		#iterate!(bt, iterations_players, vcat(day_games.P1, day_games.P2))
        iterate!(bt, iterations_all)
    end
    return predictions
end

function predict(bt::BradleyTerry, P1::Int64, P2::Int64)
	#Generate standard Bradley Terry probability for P1 winning a game
	return get(bt.rating, P1, bt.default_rating) / (get(bt.rating, P1, bt.default_rating) + get(bt.rating, P2, bt.default_rating))
end

function check_ford(bt::BradleyTerry)
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

function bt_predictable_games(bt, games)
    unratedplayers = check_ford(bt)
    println(length((unratedplayers)))
    Player1history = map(x -> x ∉ unratedplayers, games.P1)
    Player2history = map(x -> x ∉ unratedplayers, games.P2)
    return Player1history .* Player2history
end
