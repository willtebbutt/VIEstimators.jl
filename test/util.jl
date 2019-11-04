using Zygote

@testset "util" begin
    @testset "to_positive_definite" begin
        rng = MersenneTwister(123456)
        D = 11
        A = randn(rng, D, D)
        U = UpperTriangular(A)
        V = to_positive_definite(U)

        @test all(diag(V) .> 0)

        adjoint_test(
            A->to_positive_definite(UpperTriangular(A)),
            UpperTriangular(randn(rng, D, D)),
            A,
        )
    end
    @testset "to_positive_definite_softplus" begin
        rng = MersenneTwister(123456)
        D = 11
        A = randn(rng, D, D)
        U = UpperTriangular(A)
        V = to_positive_definite_softplus(U)

        @test all(diag(V) .> 0)

        adjoint_test(
            A->to_positive_definite_softplus(UpperTriangular(A)),
            UpperTriangular(randn(rng, D, D)),
            A,
        )
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
end
