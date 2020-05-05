# PairedComparisons

A package implementing methods of rating entities based on win-loss comparisons between them. At present, it contains implementations of:

- Elo
- Glicko (Mark Glickman)
- Bradley-Terry
- Whole History Rating (Remi Coulom)

The package allows each algorithm to be fitted to sets of comparisons and results provided in a DataFrame format, with or without timings specified and then to predict the results of match-ups whether they have happened before or not.

For each algorithm, the package offers a struct (Elo, Glicko, TS, BT, WHR) to record the state of the paired comparison. This is built from the parameters of the algorithm, and the ratings assigned to each player.

##Incremental and Global Algorithms

There are two separate approaches to paired comparisons that are used in this package:

###Incremental algorithms

(Elo, Glicko)

Incremental algorithms iterate through the paired comparisons one by one, storing a very small amount of information for each player (one or two numbers) and updating this information with each matching. The archetypal example of this type is the Elo algorithm, which stores only one number for each entity (their latest Elo score), and updates it depending on the comparison results with other.

This package provides three main functions to work with incremental algorithms:

- fit!(Elo, games)
- predict(Elo, player1, player2)

Example to start using an incremental algorithm:


###Global algorithms

(Bradley-Terry, Whole History Rating)

Global algorithms take account of all paired comparisons in a single sweep, and typically optimise a rating for each entity (or series of time-varying ratings for each entity) in such a way that the fit is optimised. The information carried forward and is richer, typically relying on an object that encodes the entire history of every entity being rated. They tend to require iteration to convergence, and are generally more computationally intensive.

This package provides slightly different functions to work with global algorithms:

- add_game(BT, games)
- iterate!(BT, ...)
- predict(BT, player1, player 2, ...)

###

Both
