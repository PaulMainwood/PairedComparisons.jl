# PairedComparisons

A Julia package implementing methods of rating entities based on win-loss comparisons between them. At present, it contains implementations of:

- Elo (Arpad Elo's system: https://en.wikipedia.org/wiki/Elo_rating_system)
- Glicko (Mark Glickman's system: http://www.glicko.net/research/glicko.pdf)
- Bradley-Terry (Zermelo's system, badly named: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
- Whole History Rating (Remi Coulom's time-drifting version of Bradley-Terry, https://www.remi-coulom.fr/WHR/WHR.pdf)

The package allows each algorithm to be fitted to sets of comparisons and results provided in a DataFrame format, with or without timings specified and then to predict the results of match-ups whether they have happened before or not.

For each algorithm, the package offers a struct (Elo, Glicko, BT, WHR) to record the current state, built from the parameters of the algorithm, and the ratings assigned to each player.

## Usage

All the algorithms take games in DataFrames format. They expect four or five columns (depending on whether the time of the match-up is included or not. The package expects the columns, in order:

* P1 = A unique identifier for Player 1 - the package expects an integer
* P2 = Same for Player 2
* P1Won = Number of games/points won by P1 on that day/match
* P2Won = Number of games/points won by P2 on that day/match
* Day = Integer giving a time period in which the match-ups took place (e.g., the number of day or week).

The names given to these columns do not matter, but the package expects them in the order above.

If you omit the "Day" column, there is a special "fast" version of the elo algorithm that treats each line as a separate time period and rates in the order given in the dataframe. The other algorithms need a time period and will error out if it is missed out.

So in the below, games1 is valid, though boring input for the Elo alogithm, and games2 is needed for the others.

```
games1 = DataFrame(Player1 = [1], Player2 = [2], Player1Wins = [1], Player2Wins = [0])
games2 = copy(games1)
games2.Day = [1]
```


## Incremental and Global Algorithms

There are two separate approaches to paired comparisons in this package:

### Incremental algorithms

(Elo, Glicko)

Incremental algorithms iterate through the paired comparisons one by one, storing a very small amount of information for each player (one or two numbers) and updating this information with each matching. The archetypal example of this type is the Elo algorithm, which stores only one number for each entity (their latest Elo score), and updates it depending on the comparison results with other.

This package provides three main functions to work with incremental algorithms: fit! and predict.

Example to start using an incremental algorithm:
```
elo = Elo(kfac = 85.0, default_rating = 1500.0)  # A convenience function to start with an empty dictionary. Elo() provides some default values
fit!(elo, games1) # Fit elo ratings given the games1 DataFrame (from above)
predict(elo, 1, 2) # Give a probability for player 1 to beat player 2
```
Glicko works exactly the same, but will need the DataFrame games2 as Glicko relies on ratings periods to work.

### Global algorithms

(Bradley-Terry, Whole History Rating)

Global algorithms take account of all paired comparisons in a single sweep, and typically optimise a rating for each entity (or series of time-varying ratings for each entity) in such a way that the fit is optimised. The information carried forward and is richer, typically relying on an object that encodes the entire history of every entity being rated. They tend to require iteration to convergence, and are generally more computationally intensive.

This package provides slightly different functions to work with global algorithms: add_games!, iterate!, predict
```
bt = BT(default_rating = 1.0)  # A convenience function to start with an empty dictionary. BT() provides some default values
add_games!(bt, games2) # Add the game(s) from games2 to the Dict structure of players and games inside the Bradley-Terry object
predict(bt, 1, 2) # Give a probability for player 1 to beat player 2
```

## One-ahead testing

A convenience function is provided for each function to predict the results of each time period in advance of knowing the results, then fit the algorithm to that day, then iterate again. This is helpful when assessing the predictive performance of different algorithms. Typical use looks like this:

```
elo = Elo()
fit!(elo, training_games)
one_ahead!(elo, testing_games)
```
Output is a vector of probabilities, all predicted one time period ahead, for all the games provided in the testing_games data frame.
