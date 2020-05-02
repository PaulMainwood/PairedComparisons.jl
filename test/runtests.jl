using PairedComparisons
using Test
using DataFrames

@testset "PairedComparisons.jl" begin
    # Elo tests
    elo = Elo(25.0, Dict(), 1500.0) # A standard Elo with k of 25 and default rating of 1500

    @test predict(elo, 1, 2) == 0.5

    games1 = DataFrame(Player1 = [1], Player2 = [2], Player1Wins = [1], Player2Wins = [0])
    games2 = copy(games1)
    games2.Day = [1]

    fit!(elo, games1)

    @test length(elo.ratings) == 2
    @test predict(elo, 1, 2) - 0.5359 < 0.001

    #Glicko tests
    glicko = Glicko(7.6, (1500, 250), Dict())

    @test predict(glicko, 1, 2) == 0.5

    fit!(glicko, games2)
    @test length(glicko.ratings) == 2
    @test predict(glicko, 1, 2) - 0.77594 < 0.001


end
