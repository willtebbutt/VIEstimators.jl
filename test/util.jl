using Zygote, StatsFuns
using VIEstimators: chol

@testset "util" begin
    to_test = [
        to_positive_definite,
        to_positive_definite_softplus,
        to_positive_definite_square,
    ]
    @testset "$to_psd" for to_psd in to_test
        rng = MersenneTwister(123456)
        D = 11
        A = randn(rng, D, D)
        U = UpperTriangular(A)
        V = to_psd(U)

        @test all(diag(V) .> 0)

        adjoint_test(
            A->to_psd(UpperTriangular(A)),
            UpperTriangular(randn(rng, D, D)),
            A,
        )

        if to_psd === to_positive_definite_softplus
            @test to_positive_definite_softplus_inv(V) ≈ U

            ΔV = randn(rng, D, D)
            ΔV[diagind(ΔV)] .= softplus.(ΔV[diagind(ΔV)])
            frule_test(to_positive_definite_softplus_inv, (V, ΔV); fdm=forward_fdm(5, 1))
        end
    end
    @testset "upper_triangular" begin
        rng = MersenneTwister(123456)
        D = 13
        A = randn(rng, D, D)
        U = upper_triangular(A)

        @test U == UpperTriangular(A)
        @test first(Zygote.gradient(A->sum(upper_triangular(A)), A)) isa Matrix
        adjoint_test(upper_triangular, randn(rng, D, D), A)
    end
    @testset "chol" begin
        rng = MersenneTwister(123456)
        D = 2

        # Check correctness of chol. Should be trivial.
        U_A = to_positive_definite_softplus(UpperTriangular(randn(rng, D, D)))
        A = Symmetric(U_A'U_A)
        @test chol(A) ≈ cholesky(A).U

        # Check correctness of frule for chol.
        ΔU_A = to_positive_definite_softplus(UpperTriangular(randn(rng, D, D)))
        ΔA = Symmetric(ΔU_A'ΔU_A)
        frule_test(chol, (A, ΔA); fdm=forward_fdm(5, 1))
    end
end
