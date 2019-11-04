using Random, Zygote, LinearAlgebra

@testset "elbo_estimators" begin
    # Define simple variational inference problem with tractable target distribution
    rng = MersenneTwister(123456)

    # Constructor for q distribution in terms of variational parameters.
    q = ϕ -> begin
        U = to_positive_definite_softplus(upper_triangular(ϕ.A))
        chol_Σ = Cholesky(U, 'U', 0)
        return Gaussian(ϕ.μ, chol_Σ)
    end

    # Function to compute ratio of approx to target.
    D = 5
    μ_p = randn(rng, D)
    A_p = randn(rng, D, D)
    U_p = to_positive_definite_softplus(upper_triangular(A_p))
    chol_Σ_p = Cholesky(U_p, 'U', 0)
    function log_π_over_q(ϕ, z)
        log_π̃ = -sum(abs2.(chol_Σ_p.U' \ (z - μ_p))) / 2
        log_q = logpdf(q(ϕ), z)
        return log_π̃ - log_q
    end

    # Verify that the RPT∇ (and it's gradient) is correct in expectation when q is tight.
    @testset "RPT∇" begin
        ϕ = (μ=μ_p, A=A_p)
        results = [STL∇(rng, q, ϕ, log_π_over_q) for _ in 1:100_000]
        elbo = mean(first, results)
        ∂μ = mean(x->last(x).μ, results)
        ∂A = mean(x->last(x).A, results)

        @test elbo ≈ (logdet(chol_Σ_p) + D * log(2π)) / 2 rtol=1e-2 atol=1e-2
        @test ∂μ ≈ zeros(size(∂μ)) rtol=1e-1 atol=1e-1
        @test ∂A ≈ zeros(size(∂A)) rtol=1e-1 atol=1e-1
    end

    # Verify that gradient is zero using STL estimator when approximation tight.
    @testset "STL∇" begin
        ϕ = (μ=μ_p, A=A_p)
        elbo, ∂ϕ, ∂ϕ_score = STL∇(rng, q, ϕ, log_π_over_q)

        @test elbo ≈ (logdet(chol_Σ_p) + D * log(2π)) / 2
        @test all(∂ϕ.μ .≈ 0)
        @test all(∂ϕ.A .≈ 0)
    end
end
