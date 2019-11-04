using Distributions

@testset "gaussian" begin
    @testset "Gaussian" begin
        rng = MersenneTwister(123456)
        D = 11
        S = 7
        m = randn(rng, D)
        A = randn(rng, D, D)
        Σ = Symmetric(A * A' + I)

        distributions_gaussian = MvNormal(m, Σ)
        my_gaussian = Gaussian(m, cholesky(Σ))

        X = rand(rng, distributions_gaussian, S)
        @test logpdf(distributions_gaussian, X) ≈ logpdf(my_gaussian, X)
    end
    @testset "normal_logpdf" begin
        rng = MersenneTwister(123456)
        P, Q = 128, 256
        Y = randn(rng, P, Q)
        M = randn(rng, P, Q)
        σs = exp.(randn(rng, P, Q)) .+ 1e-6
        @test normal_logpdf(Y, M, σs) ≈ logpdf.(Normal.(M, σs), Y)
    end
    @testset "GaussianPair" begin
        rng = MersenneTwister(123456)

        # First Gaussian.
        D1 = 11
        m1 = randn(rng, D1)
        A1 = UpperTriangular(randn(rng, D1, D1))
        Σ1 = Cholesky(to_positive_definite_softplus(A1), 'U', 0)
        X1 = Gaussian(m1, Σ1)

        # Second Gaussian.
        D2 = 7
        m2 = randn(rng, D2)
        A2 = UpperTriangular(randn(rng, D2, D2))
        Σ2 = Cholesky(to_positive_definite_softplus(A2), 'U', 0)
        X2 = Gaussian(m2, Σ2)

        # Create a Gaussian pair.
        X = GaussianPair(X1, X2)

        # Check that rand works.
        S = 999
        y = rand(rng, X)
        Y = rand(rng, X, S)

        # Check that logpdf is consistent with independent Gaussians.
        @test length(X) == length(X1) + length(X2)
        @test logpdf(X, y) ≈ logpdf(X.x1, y.x1) + logpdf(X.x2, y.x2)
        @test logpdf(X, Y) ≈ logpdf(X.x1, Y.x1) + logpdf(X.x2, Y.x2)
    end
end