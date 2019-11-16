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
    @testset "StandardGaussian" begin
        rng = MersenneTwister(123456)
        D = 3

        X = StandardGaussian(D)
        @test length(X) == 3

        # Ensure the logpdf is the same as a regular old Gaussian.
        X_regular = Gaussian(zeros(D), cholesky(Matrix{Float64}(I, D, D)))

        x = rand(rng, X_regular)
        @test logpdf(X, x) ≈ logpdf(X_regular, x)

        xs = rand(rng, X_regular, 5)
        @test logpdf(X, xs) ≈ logpdf(X_regular, xs)

        # Ensure that sampling behaviour is also roughly the same.
        @test rand(MersenneTwister(123456), X) ≈ rand(MersenneTwister(123456), X_regular)
        @test rand(MersenneTwister(123456), X, 5) ≈ rand(MersenneTwister(123456), X_regular, 5)
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
    @testset "natural parameters" begin
        rng = MersenneTwister(123456)
        D = 3
        m = randn(rng, D)
        A = randn(rng, D, D)
        S = A'A + I

        θ₁, θ₂ = standard_to_natural(m, S)
        m′, S′ = natural_to_standard(θ₁, θ₂)

        @test m ≈ m′
        @test S ≈ S′

        Δθ₁ = randn(rng, D)

        A_ = randn(rng, D, D)
        Δθ₂ = A_ * A_' + I
        frule_test(natural_to_standard, (θ₁, Δθ₁), (θ₂, Δθ₂); fdm=backward_fdm(5, 1))
    end
    @testset "expectation parameters" begin
        rng = MersenneTwister(123456)
        D = 3
        m = randn(rng, D)
        A = randn(rng, D, D)
        S = A'A + I

        θ₁, θ₂ = standard_to_expectation(m, S)
        m′, S′ = expectation_to_standard(θ₁, θ₂)

        @test m ≈ m′
        @test S ≈ S′
    end
    @testset "expectation / natural parameters conversions" begin
        rng = MersenneTwister(123456)
        D = 3
        m = randn(rng, D)
        A = randn(rng, D, D)
        S = A'A + I

        @testset "natural to expectation" begin
            θ₁, θ₂ = standard_to_natural(m, S)
            η₁, η₂ = natural_to_expectation(θ₁, θ₂)
            m′, S′ = expectation_to_standard(η₁, η₂)
            @test m ≈ m′
            @test S ≈ S′
        end

        @testset "expectation to natural" begin
            η₁, η₂ = standard_to_expectation(m, S)
            θ₁, θ₂ = expectation_to_natural(η₁, η₂)
            m′, S′ = natural_to_standard(θ₁, θ₂)
            @test m ≈ m′
            @test S ≈ S′
        end
    end
    @testset "unconstrained" begin
        rng = MersenneTwister(123456)
        D = 2
        m = randn(rng, D)
        U = UpperTriangular(randn(rng, D, D))

        @testset "standard is psd" begin
            _, S = unconstrained_to_standard(m, U)
            @test all(eigvals(S) .> 0)
        end
        @testset "standard_to_unconstrained" begin

            # Ensure inverse is correct.
            ms, Ss = unconstrained_to_standard(m, U)
            m′, U′ = standard_to_unconstrained(ms, Ss)
            @test m ≈ m′
            @test U ≈ U′

            # Use positive definite forwards sensitivity to test Ss.
            Δms = randn(rng, length(ms))
            _ΔSs = randn(rng, size(Ss))
            ΔSs = _ΔSs * _ΔSs' + I

            fdm = forward_fdm(5, 1)
            frule_test(standard_to_unconstrained, (ms, Δms), (Ss, ΔSs); fdm=fdm)
        end
        @testset "natural is nsd" begin
            θ₁, θ₂ = unconstrained_to_natural(m, U)
            @test all(eigvals(θ₂) .< 0)
        end
        @testset "natural_to_unconstrained" begin
            m′, U′ = natural_to_unconstrained(unconstrained_to_natural(m, U)...)
            @test m ≈ m′
            @test U ≈ U′

            θ₁, θ₂ = unconstrained_to_natural(m, U)
            θ₁_comp, θ₂_comp = standard_to_natural(unconstrained_to_standard(m, U)...)
            @test θ₁_comp ≈ θ₁
            @test θ₂_comp ≈ θ₂

            Δθ₁ = randn(rng, length(θ₁))
            Δθ₂_ = randn(rng, size(θ₂))
            Δθ₂ = Δθ₂_ * Δθ₂_'
            fdm = backward_fdm(5, 1)
            frule_test(natural_to_unconstrained, (θ₁, Δθ₁), (θ₂, Δθ₂); fdm=fdm)
        end
        @testset "expectation is psd" begin
            _, S = unconstrained_to_expectation(m, U)
            @test all(eigvals(S) .> 0)
        end
        @testset "expectation_to_unconstrained" begin
            m′, U′ = expectation_to_unconstrained(unconstrained_to_expectation(m, U)...)
            @test m ≈ m′
            @test U ≈ U′

            η₁, η₂ = unconstrained_to_expectation(m, U)
            η₁_comp, η₂_comp = standard_to_expectation(unconstrained_to_standard(m, U)...)
            @test η₁ ≈ η₁_comp
            @test η₂ ≈ η₂_comp
        end
    end
end
