using PairedComparisons
using Test
using DataFrames

@testset "PairedComparisons.jl" begin
    # Elo tests
    elo = Elo(kfac = 25.0) # A standard Elo with k of 25 and default rating of 1500

    @test predict(elo, 1, 2) == 0.5

    games1 = DataFrame(P1 = [1], P2 = [2], P1_wins = [1], P2_wins = [0])
    games2 = copy(games1)
    games2.Period = [1]

    fit!(elo, games1)

    @test length(elo.ratings) == 2
    @test abs(predict(elo, 1, 2) - 0.5359159269451023) < 0.001

    #Glicko tests
    glicko = Glicko(c = 85.0, default_rating = (1500.0, 350.0, -1))

    @test predict(glicko, 1, 2; rating_day=1) == 0.5

    fit!(glicko, games2)
    @test length(glicko.ratings) == 2
    @test abs(predict(glicko, 1, 2; rating_day=1) - 0.72720) < 0.001

    #BradleyTerry tests
    bt = BT()

    @test predict(bt, 1, 2) == 0.5
    add_games!(bt, games2)
    @test length(bt.playergames) == 2
    @test abs(loglikelihood(bt) + 2.0794) < 0.001
    iterate!(bt, 50)
    @test abs(loglikelihood(bt) + 1.6847756) < 0.001
    @test abs(predict(bt, 1, 2)  - 0.7419441266134) < 0.001

    #WHR tests

    whr = WHR()
    add_games!(whr, games2)
    @test length(whr.playerdaygames) == 2
    @test abs(predict(whr, 1, 2) - 0.5) < 0.0001
    @test abs(loglikelihood(whr) + 4.63934111119) < 0.001
    iterate!(whr, 5)
    @test abs(predict(whr, 1, 2) - 0.684114747) < 0.001


end
