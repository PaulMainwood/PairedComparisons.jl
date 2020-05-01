# PairedComparisons

A package implementing methods of rating entities based on win-loss comparisons between them. At present, it contains implementations of:

- Elo
- Glicko (Mark Glickman)
- Trueskill (Paired version only)
- Bradley-Terry
- Whole History Rating (Remi Coulom)

The package allows each algorithm to be fitted to sets of comparisons and results provided in a DataFrame format, with or without timings specified and then to predict the results of match-ups whether they have happened before or not.
