using Distributions, Zygote

@testset "gaussian" begin
    # @testset "Gaussian" begin
    #     rng = MersenneTwister(123456)
    #     D = 11
    #     S = 7
    #     m = randn(rng, D)
    #     A = randn(rng, D, D)
    #     Σ = Symmetric(A * A' + I)

    #     distributions_gaussian = MvNormal(m, Σ)
    #     my_gaussian = Gaussian(m, cholesky(Σ))

    #     X = rand(rng, distributions_gaussian, S)
    #     @test logpdf(distributions_gaussian, X) ≈ logpdf(my_gaussian, X)
    # end
    # @testset "normal_logpdf" begin
    #     rng = MersenneTwister(123456)
    #     P, Q = 128, 256
    #     Y = randn(rng, P, Q)
    #     M = randn(rng, P, Q)
    #     σs = exp.(randn(rng, P, Q)) .+ 1e-6
    #     @test normal_logpdf(Y, M, σs) ≈ logpdf.(Normal.(M, σs), Y)
    # end
    # @testset "StandardGaussian" begin
    #     rng = MersenneTwister(123456)
    #     D = 3

    #     X = StandardGaussian(D)
    #     @test length(X) == 3

    #     # Ensure the logpdf is the same as a regular old Gaussian.
    #     X_regular = Gaussian(zeros(D), cholesky(Matrix{Float64}(I, D, D)))

    #     x = rand(rng, X_regular)
    #     @test logpdf(X, x) ≈ logpdf(X_regular, x)

    #     xs = rand(rng, X_regular, 5)
    #     @test logpdf(X, xs) ≈ logpdf(X_regular, xs)

    #     # Ensure that sampling behaviour is also roughly the same.
    #     @test rand(MersenneTwister(123456), X) ≈ rand(MersenneTwister(123456), X_regular)
    #     @test rand(MersenneTwister(123456), X, 5) ≈ rand(MersenneTwister(123456), X_regular, 5)
    # end
    # @testset "GaussianPair" begin
    #     rng = MersenneTwister(123456)

    #     # First Gaussian.
    #     D1 = 11
    #     m1 = randn(rng, D1)
    #     A1 = UpperTriangular(randn(rng, D1, D1))
    #     Σ1 = Cholesky(to_positive_definite_softplus(A1), 'U', 0)
    #     X1 = Gaussian(m1, Σ1)

    #     # Second Gaussian.
    #     D2 = 7
    #     m2 = randn(rng, D2)
    #     A2 = UpperTriangular(randn(rng, D2, D2))
    #     Σ2 = Cholesky(to_positive_definite_softplus(A2), 'U', 0)
    #     X2 = Gaussian(m2, Σ2)

    #     # Create a Gaussian pair.
    #     X = GaussianPair(X1, X2)

    #     # Check that rand works.
    #     S = 999
    #     y = rand(rng, X)
    #     Y = rand(rng, X, S)

    #     # Check that logpdf is consistent with independent Gaussians.
    #     @test length(X) == length(X1) + length(X2)
    #     @test logpdf(X, y) ≈ logpdf(X.x1, y.x1) + logpdf(X.x2, y.x2)
    #     @test logpdf(X, Y) ≈ logpdf(X.x1, Y.x1) + logpdf(X.x2, Y.x2)
    # end
    # @testset "natural parameters" begin
    #     rng = MersenneTwister(123456)
    #     D = 3
    #     m = randn(rng, D)
    #     A = randn(rng, D, D)
    #     S = A'A + I

    #     θ₁, θ₂ = standard_to_natural(m, S)
    #     m′, S′ = natural_to_standard(θ₁, θ₂)

    #     @test m ≈ m′
    #     @test S ≈ S′

    #     Δθ₁ = randn(rng, D)

    #     A_ = randn(rng, D, D)
    #     Δθ₂ = A_ * A_' + I
    #     frule_test(natural_to_standard, (θ₁, Δθ₁), (θ₂, Δθ₂); fdm=backward_fdm(5, 1))
    # end
    # @testset "expectation parameters" begin
    #     rng = MersenneTwister(123456)
    #     D = 3
    #     m = randn(rng, D)
    #     A = randn(rng, D, D)
    #     S = A'A + I

    #     θ₁, θ₂ = standard_to_expectation(m, S)
    #     m′, S′ = expectation_to_standard(θ₁, θ₂)

    #     @test m ≈ m′
    #     @test S ≈ S′
    # end
    # @testset "expectation / natural parameters conversions" begin
    #     rng = MersenneTwister(123456)
    #     D = 3
    #     m = randn(rng, D)
    #     A = randn(rng, D, D)
    #     S = A'A + I

    #     @testset "natural to expectation" begin
    #         θ₁, θ₂ = standard_to_natural(m, S)
    #         η₁, η₂ = natural_to_expectation(θ₁, θ₂)
    #         m′, S′ = expectation_to_standard(η₁, η₂)
    #         @test m ≈ m′
    #         @test S ≈ S′
    #     end

    #     @testset "expectation to natural" begin
    #         η₁, η₂ = standard_to_expectation(m, S)
    #         θ₁, θ₂ = expectation_to_natural(η₁, η₂)
    #         m′, S′ = natural_to_standard(θ₁, θ₂)
    #         @test m ≈ m′
    #         @test S ≈ S′
    #     end
    # end
    @testset "unconstrained" begin
        rng = MersenneTwister(123456)
        D = 3
        m = randn(rng, D)
        U = UpperTriangular(randn(rng, D, D))

        # @testset "standard is psd" begin
        #     _, S = unconstrained_to_standard(m, U)
        #     @test all(eigvals(S) .> 0)
        # end
        # @testset "standard_to_unconstrained" begin

        #     # Ensure inverse is correct.
        #     ms, Ss = unconstrained_to_standard(m, U)
        #     m′, U′ = standard_to_unconstrained(ms, Ss)
        #     @test m ≈ m′
        #     @test U ≈ U′

        #     # Use positive definite forwards sensitivity to test Ss.
        #     Δms = randn(rng, length(ms))
        #     _ΔSs = randn(rng, size(Ss))
        #     ΔSs = _ΔSs * _ΔSs' + I

        #     fdm = forward_fdm(5, 1)
        #     frule_test(standard_to_unconstrained, (ms, Δms), (Ss, ΔSs); fdm=fdm)
        # end
        # @testset "natural is nsd" begin
        #     θ₁, θ₂ = unconstrained_to_natural(m, U)
        #     @test all(eigvals(θ₂) .< 0)
        # end
        # @testset "natural_to_unconstrained" begin
        #     m′, U′ = natural_to_unconstrained(unconstrained_to_natural(m, U)...)
        #     @test m ≈ m′
        #     @test U ≈ U′

        #     θ₁, θ₂ = unconstrained_to_natural(m, U)
        #     θ₁_comp, θ₂_comp = standard_to_natural(unconstrained_to_standard(m, U)...)
        #     @test θ₁_comp ≈ θ₁
        #     @test θ₂_comp ≈ θ₂

        #     Δθ₁ = randn(rng, length(θ₁))
        #     Δθ₂_ = randn(rng, size(θ₂))
        #     Δθ₂ = Δθ₂_ * Δθ₂_'
        #     fdm = backward_fdm(5, 1)
        #     frule_test(natural_to_unconstrained, (θ₁, Δθ₁), (θ₂, Δθ₂); fdm=fdm)
        # end
        # @testset "expectation is psd" begin
        #     _, S = unconstrained_to_expectation(m, U)
        #     @test all(eigvals(S) .> 0)
        # end
        # @testset "expectation_to_unconstrained" begin
        #     m′, U′ = expectation_to_unconstrained(unconstrained_to_expectation(m, U)...)
        #     @test m ≈ m′
        #     @test U ≈ U′

        #     η₁, η₂ = unconstrained_to_expectation(m, U)
        #     η₁_comp, η₂_comp = standard_to_expectation(unconstrained_to_standard(m, U)...)
        #     @test η₁ ≈ η₁_comp
        #     @test η₂ ≈ η₂_comp
        # end
        # @testset "natural gradient tests" begin
        #     rng = MersenneTwister(123456)
        #     D = 3
        #     η̄₁, η̄₂ = randn(rng, D), randn(rng, D, D)
        #     η̄ = hcat(η̄₁, η̄₂)

        #     m, U = randn(rng, D), UpperTriangular(randn(rng, D, D))
        #     _, S = unconstrained_to_standard(m, U)
        #     θ₁, θ₂ = unconstrained_to_natural(m, U)

        #     adjoint_test(
        #         (θ₁, θ₂) -> hcat(natural_to_expectation(θ₁, θ₂)...),
        #         η̄, θ₁, θ₂,
        #     )

        #     adjoint_test(
        #         (m, S) -> hcat(standard_to_expectation(m, S)...),
        #         η̄, m, S,
        #     )

        #     adjoint_test(
        #         (m, U) -> hcat(unconstrained_to_standard(m, UpperTriangular(U))...),
        #         η̄, m, U,
        #     )

        #     adjoint_test(
        #         U -> Matrix(UpperTriangular(U)'UpperTriangular(U)),
        #         randn(rng, D, D), randn(rng, D, D),
        #     )

        #     adjoint_test(
        #         (m, U) -> hcat(unconstrained_to_expectation(m, UpperTriangular(U))...),
        #         η̄, m, U,
        #     )

        #     # Compute sensible norm in the first way.
        #     _, back_nat_to_exp = Zygote.forward(natural_to_expectation, θ₁, θ₂)
        #     θ̄₁, θ̄₂ = back_nat_to_exp((η̄₁, η̄₂))
        #     display(hcat(θ̄₁, θ̄₂))
        #     println()

        #     norm_1 = dot(θ̄₁, η̄₁) + dot(θ̄₂, η̄₂)
        #     @show dot(θ̄₁, η̄₁), dot(θ̄₂, η̄₂)
        #     @show norm_1

        #     # Compute sensible norm in the second way.
        #     _, back_unconstrained_to_expectation = Zygote.forward(
        #         unconstrained_to_expectation, m, U,
        #     )
        #     (m̄, Ū) = back_unconstrained_to_expectation((η̄₁, η̄₂))

        #     _, pushforward_nat_to_uncon = frule(natural_to_unconstrained, θ₁, θ₂)
        #     Δm_nat, ΔU_nat = pushforward_nat_to_uncon(DoesNotExist(), η̄₁, η̄₂)

        #     @show dot(m̄, Δm_nat), dot(Ū, ΔU_nat)
        #     norm_2 = dot(m̄, Δm_nat) + dot(Ū, ΔU_nat)
        #     @show norm_2
        # end
        @testset "positive definite tests" begin
            rng = MersenneTwister(123456)
            D = 2

            function expectation_to_natural_vec(η)
                η₁ = η[1:D]
                η₂ = reshape(η[D+1:end], D, D)
                θ₁, θ₂ = expectation_to_natural(η₁, η₂)
                return vec(hcat(θ₁, θ₂))
            end

            function natural_to_expectation_vec(θ)
                θ₁ = θ[1:D]
                θ₂ = reshape(θ[D+1:end], D, D)
                η₁, η₂ = natural_to_expectation(θ₁, θ₂)
                return vec(hcat(η₁, η₂))
            end

            # Construct natural parameters and convert to vector.
            m = randn(rng, D)
            U = UpperTriangular(randn(rng, D, D))
            θ₁, θ₂ = unconstrained_to_natural(m, U)
            θ = vec(hcat(θ₁, θ₂))
            η = natural_to_expectation_vec(θ)

            m, S = unconstrained_to_standard(m, U)
            θ_std = vec(hcat(m, S))

            println("vals")
            display(θ₁)
            println()
            display(θ₂)
            println()

            function expectation_mat_vec(θ₂)
                _, η₂ = natural_to_expectation(zeros(D), reshape(θ₂, D, D))
                return vec(η₂)
            end

            function natural_to_standard_vec(θ)
                m = θ[1:D]
                S = reshape(θ[D+1:end], D, D)
                η₁, η₂ = Zygote.@showgrad(natural_to_standard(m, S))
                return vec(hcat(η₁, η₂))
            end

            function standard_to_expectation_vec(θ)
                θ₁ = θ[1:D]
                θ₂ = reshape(θ[D+1:end], D, D)
                η₁, η₂ = Zygote.@showgrad(standard_to_expectation(θ₁, θ₂))
                return vec(hcat(η₁, η₂))
            end

            J = Matrix{Float64}(undef, length(θ), length(θ))
            Jinv = Matrix{Float64}(undef, length(θ), length(θ))
            for d in eachindex(θ)

                println("d = $d")


                # Define dth seed for reverse-pass.
                Δη = zeros(length(θ))
                Δη[d] = 1.0

                # Compute dth column of J
                _, back = Zygote.forward(natural_to_expectation_vec, θ)
                Δθ = first(back(Δη))
                J[:, d] .= Δθ

                _, back = Zygote.forward(expectation_to_natural_vec, η)
                Jinv[:, d] = first(back(Δη))

                println()
                println()
                println()
            end

            display(eigvals(Symmetric(Matrix(LowerTriangular(J)))))
            println()

            println("J'")
            display(J')
            println()

            println("Jinv")
            display(Jinv)
            println()

            display(eigvals(Jinv))
            println()

            display(J - inv(Jinv))
            println()

            


            v = randn(length(θ))
            _, back = Zygote.forward(natural_to_expectation_vec, θ)
            @test first(back(v))'v ≈ v' * (J * v)

            @test J ≈ J'
            @test all(eigvals(Symmetric(J)) .> 0)

            # J_mat = Matrix{Float64}(undef, length(θ₂), length(θ₂))
            # for d in eachindex(θ₂)

            #     # Define dth seed for reverse-pass.
            #     Δη = zeros(length(θ₂))
            #     Δη[d] = 1.0

            #     # Compute dth column of J
            #     _, back = Zygote.forward(expectation_mat_vec, vec(θ₂))
            #     Δθ₂ = first(back(Δη))
            #     J_mat[:, d] .= Δθ₂
            # end

            # @show "J_mat stuff"

            # display(J_mat)
            # println()

            # display(J_mat - J_mat')
            # println()

            # Check that stuff composes.

            # J_stn = Matrix{Float64}(undef, length(θ), length(θ))
            # for d in eachindex(θ)

            #     println("d = $d")


            #     # Define dth seed for reverse-pass.
            #     Δη = zeros(length(θ))
            #     Δη[d] = 1.0

            #     # Compute dth column of J
            #     _, back = Zygote.forward(natural_to_standard_vec, θ)
            #     Δθ = first(back(Δη))
            #     J_stn[:, d] .= Δθ

            #     println()
            #     println()
            #     println()
            # end

            # J_exp = Matrix{Float64}(undef, length(θ), length(θ))
            # for d in eachindex(θ)

            #     println("d = $d")


            #     # Define dth seed for reverse-pass.
            #     Δη = zeros(length(θ))
            #     Δη[d] = 1.0

            #     # Compute dth column of J
            #     _, back = Zygote.forward(standard_to_expectation_vec, θ_std)
            #     Δθ = first(back(Δη))
            #     J_exp[:, d] .= Δθ

            #     println()
            #     println()
            #     println()
            # end

            # display(J_exp')
            # println()

            # display(θ₁)
            # println()
            # display(-inv(θ₂) ./ 2)
            # println()

            # display(J_stn')
            # println()

            # for d in eachindex(θ₂)

            #     println("d = $d")
            #     Δη = zeros(length(θ₂))
            #     Δη[d] = 1.0
            #     Δη = reshape(Δη, D, D)
            #     display(- inv(θ₂) * Δη * inv(θ₂) ./ 4)
            #     println()
            # end

            # display(J_exp' * J_stn')
            # println()

            # display(J_exp' * J_stn' - J')
            # println()
        end
    end
end
