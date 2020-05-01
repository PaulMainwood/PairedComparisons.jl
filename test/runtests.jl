using PairedComparisons
using Test
using DataFrames

@testset "PairedComparisons.jl" begin
    # Write your own tests here.
    elo = Elo(25.0, Dict(), 1500.0) # A standard Elo with k of 25 and default rating of 1500

    @test predict(elo, 1, 2) == 0.5


    games1 = DataFrame(Player1 = [1], Player2 = [2], Player1Wins = [1], Player2Wins = [0])



end
