module VIEstimators

    using Random, LinearAlgebra, StatsFuns, Zygote, ZygoteRules, Distributions
    import ChainRulesCore: frule, DoesNotExist

    # Variational inference-related things.
    include("util.jl")
    include("gaussian.jl")
    include("elbo_estimators.jl")
end # module
