using Test, VIEstimators, Random, LinearAlgebra

# Helper functionality for testing, primarily checking custom adjoints.
include("test_util.jl")

@testset "VIEstimators" begin
    # include("util.jl")
    include("gaussian.jl")
    # include("elbo_estimators.jl")
end
