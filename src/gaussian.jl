import Distributions: logpdf
import Random: rand
import Base: length
export Gaussian, normal_logpdf, StandardGaussian, GaussianPair, logpdf
export standard_to_natural, natural_to_standard
export standard_to_expectation, expectation_to_standard
export expectation_to_natural, natural_to_expectation
export standard_to_unconstrained, unconstrained_to_standard
export natural_to_unconstrained, unconstrained_to_natural
export expectation_to_unconstrained, unconstrained_to_expectation



"""
    Gaussian{Tm, Tchol_Σ<:Cholesky}

A (multivariate) Gaussian with mean vector `m` and cholesky factorisation `chol_Σ` of the
covariance matrix.
"""
struct Gaussian{Tm<:AbstractVector{<:Real}, Tchol_Σ<:Cholesky}
    m::Tm
    chol_Σ::Tchol_Σ
end

length(x::Gaussian) = length(x.m)

rand(rng::AbstractRNG, x::Gaussian) = x.m + x.chol_Σ.U' * randn(rng, length(x.m))
function rand(rng::AbstractRNG, x::Gaussian, S::Int)
    return x.m .+ x.chol_Σ.U' * randn(rng, length(x.m), S)
end

# Returns a `Vector` of length `size(Y, 2)` containing the logpdf of each column of `Y`
# under `x`.
function logpdf(x::Gaussian, Y::AbstractMatrix{<:Real})
    @assert length(x) == size(Y, 1)
    tmp = length(x) * log(2π) + logdet(x.chol_Σ)
    return .-(tmp .+ diag_Xt_invA_X(x.chol_Σ, Y .- x.m)) / 2
end
logpdf(x::Gaussian, y::AbstractVector{<:Real}) = logpdf(x, reshape(y, :, 1))[1]

# A vectorised logpdf for the normal distribution because Zygote's broadcasting sucks
# if you don't do things correctly :S Even differnetiating through this kind of sucks,
# which is a real shame.
function normal_logpdf(y, m, σ)
    return .-(log(2π) .+ 2 .* log.(σ) .+ ((y .- m) ./ σ).^2) ./ 2
end



"""
    StandardGaussian

`D`-dimensional multivariate Gaussian with zero mean and identity covariance.
"""
struct StandardGaussian
    D::Int
end

length(X::StandardGaussian) = X.D

rand(rng::AbstractRNG, X::StandardGaussian) = randn(rng, length(X))
rand(rng::AbstractRNG, X::StandardGaussian, S::Int) = randn(rng, length(X), S)

function logpdf(X::StandardGaussian, y::AbstractVector{<:Real})
    return -(X.D * log(2π) + sum(abs2, y)) / 2
end

function logpdf(X::StandardGaussian, Y::AbstractMatrix{<:Real})
    return .-(X.D * log(2π) .+ vec(sum(abs2, Y;  dims=1))) / 2
end



"""
    GaussianPair{Tx1<:Gaussian, Tx2<:Gaussian}

A pair of independent Gaussians `x1` and `x2`.
"""
struct GaussianPair{Tx1<:Gaussian, Tx2<:Gaussian}
    x1::Tx1
    x2::Tx2
end

length(x::GaussianPair) = length(x.x1) + length(x.x2) 

function rand(rng::AbstractRNG, x::GaussianPair, S::Int)
    return (x1=rand(rng, x.x1, S), x2=rand(rng, x.x2, S))
end
rand(rng::AbstractRNG, x::GaussianPair) = (x1=rand(rng, x.x1), x2=rand(rng, x.x2))

function logpdf(x::GaussianPair, Y::NamedTuple{(:x1, :x2)})
    return logpdf(x.x1, Y.x1) + logpdf(x.x2, Y.x2)
end



#
# Constrained parametrisations.
#

"""
    natural_to_standard(θ₁::AbstractVector{<:Real}, θ₂::AbstractMatrix{<:Real})

Convert from natural to standard parametrisation.
"""
function natural_to_standard(θ₁::AbstractVector{<:Real}, θ₂::Symmetric{<:Real})
    S = Symmetric(inv(cholesky(-θ₂)) ./ 2)
    return S * θ₁, S
end


# forwards-mode rule for natural_to_standard. Avoids some repeated computation.
function frule(
    ::typeof(natural_to_standard),
    θ₁::AbstractVector{<:Real},
    θ₂::Symmetric{<:Real},
)
    C = cholesky(-θ₂)
    S = Symmetric(inv(C) ./ 2)
    m = S * θ₁
    function natural_to_standard_pushforward(Δself, Δθ₁, Δθ₂)
        ΔS = Symmetric((C \ Matrix((C \ Δθ₂')')) ./ 2)
        Δm = ΔS * θ₁ + S * Δθ₁
        return (Δm, ΔS)
    end
    return (m, S), natural_to_standard_pushforward
end

"""
    standard_to_natural(m::AbstractVector{<:Real}, S::Symmetric{<:Real})

Convert from standard to natural parametrisation.
"""
function standard_to_natural(m::AbstractVector{<:Real}, S::Symmetric{<:Real})
    C = inv(cholesky(S))
    return C * m, Symmetric(C ./ (-2))
end

"""
    expectation_to_standard(η₁::AbstractVector{<:Real}, η₂::Symmetric{<:Real})

Convert from expectation to standard parametrisation.
"""
function expectation_to_standard(η₁::AbstractVector{<:Real}, η₂::Symmetric{<:Real})
    return η₁, η₂ - Symmetric(η₁ * η₁')
end

"""
    standard_to_expectation(m::AbstractVector{<:Real}, S::Symmetric{<:Real})

Convert from standard to natural parametrisation.
"""
function standard_to_expectation(m::AbstractVector{<:Real}, S::Symmetric{<:Real})
    return m, Symmetric(S + m * m')
end

function frule(
    ::typeof(standard_to_expectation),
    m::AbstractVector{<:Real},
    S::Symmetric{<:Real},
)
    function standard_to_expectation_pushforward(Δself, Δm, ΔS::Symmetric{<:Real})
        Δη₁ = Δm
        Δη₂ = Symmetric(ΔS + m * Δm' + Δm * m')
        return Δη₁, Δη₂
    end
    return standard_to_expectation(m, S), standard_to_expectation_pushforward
end

"""
    expectation_to_natural(η₁::AbstractVector{<:Real}, η₂::Symmetric{<:Real})

Convert from expectation to natural parameters.
"""
function expectation_to_natural(η₁::AbstractVector{<:Real}, η₂::Symmetric{<:Real})
    return standard_to_natural(expectation_to_standard(η₁, η₂)...)
end

"""
    natural_to_expectation(θ₁::AbstractVector{<:Real}, θ₂::Symmetric{<:Real})

Convert from natural to expectation parametrisation.
"""
function natural_to_expectation(θ₁::AbstractVector{<:Real}, θ₂::Symmetric{<:Real})
    m, S = natural_to_standard(θ₁, θ₂)
    η₁, η₂ = standard_to_expectation(m, S)
    return η₁, η₂
end

function frule(
    ::typeof(natural_to_expectation),
    θ₁::AbstractVector{<:Real},
    θ₂::Symmetric{<:Real},
)
    (m, S), nat_to_std_pushforward = frule(natural_to_standard, θ₁, θ₂)
    (η₁, η₂), std_to_exp_pushforward = frule(standard_to_expectation, m, S)
    function natural_to_expectation_pushforward(Δself, Δθ₁, Δθ₂::Symmetric)
        Δm, ΔS = nat_to_std_pushforward(DoesNotExist(), Δθ₁, Δθ₂)
        return std_to_exp_pushforward(DoesNotExist(), Δm, ΔS)
    end
    return (η₁, η₂), natural_to_expectation_pushforward
end



#
# Unconstrained parametrisations.
#

"""
    standard_to_unconstrained(m::AbstractVector{<:Real}, S::Symmetric{<:Real})

Convert from standard to mean and unconstrained upper-triangular matrix.
"""
function standard_to_unconstrained(m::AbstractVector{<:Real}, S::Symmetric{<:Real})
    return m, to_positive_definite_softplus_inv(chol(S))
end

function frule(
    ::typeof(standard_to_unconstrained),
    m::AbstractVector{<:Real},
    S::Symmetric{<:Real},
)
    # Perform forwards-pass and retain information required to compute pushforward.
    U, pushforward_U = frule(chol, S)
    V, pushforward_V = frule(to_positive_definite_softplus_inv, U)

    function standard_to_unconstrained_pushforward(Δself, Δm, ΔS::Symmetric{<:Real})
        ΔU = pushforward_U(DoesNotExist(), ΔS)
        ΔV = pushforward_V(DoesNotExist(), ΔU)
        return (Δm, ΔV)
    end

    return (m, V), standard_to_unconstrained_pushforward
end

"""
    unconstrained_to_standard(m::AbstractVector{<:Real}, U::UpperTriangular{<:Real})

Convert from mean + unconstrained upper-triangular matrix to standard parametrisation.
"""
function unconstrained_to_standard(m::AbstractVector{<:Real}, U::UpperTriangular{<:Real})
    UA = to_positive_definite_softplus(U)
    return m, Symmetric(UA'UA)
end


"""
    natural_to_unconstrained(θ₁::AbstractVector{<:Real}, θ₂::AbstractMatrix{<:Real})

Convert from natural to mean and unconstrained upper-triangular matrix.
"""
function natural_to_unconstrained(θ₁::AbstractVector{<:Real}, θ₂::Symmetric{<:Real})
    return standard_to_unconstrained(natural_to_standard(θ₁, θ₂)...)
end

function frule(
    ::typeof(natural_to_unconstrained),
    θ₁::AbstractVector{<:Real},
    θ₂::Symmetric{<:Real},
)
    (m_, S), pushforward_standard = frule(natural_to_standard, θ₁, θ₂)
    (m, U), pushforward_unconstrained = frule(standard_to_unconstrained, m_, S)

    function pushforward_natural_to_unconstrained(Δself, Δθ₁, Δθ₂::Symmetric{<:Real})
        Δm_, ΔS = pushforward_standard(DoesNotExist(), Δθ₁, Δθ₂)
        return pushforward_unconstrained(DoesNotExist(), Δm_, ΔS)
    end
    return (m, U), pushforward_natural_to_unconstrained
end

"""
    unconstrained_to_natural(m::AbstractVector{<:Real}, U::AbstractMatrix{<:Real})

Convert from mean and unconstrained upper-triangular matrix to natural.
"""
function unconstrained_to_natural(m::AbstractVector{<:Real}, U::UpperTriangular{<:Real})
    return standard_to_natural(unconstrained_to_standard(m, U)...)
end

"""
    expectation_to_unconstrained(η₁::AbstractVector{<:Real}, η₂::AbstractMatrix{<:Real})

Convert from expecation parameters to mean and unconstrained upper-triangular matrix.
"""
function expectation_to_unconstrained(
    η₁::AbstractVector{<:Real},
    η₂::AbstractMatrix{<:Real},
)
    return standard_to_unconstrained(expectation_to_standard(η₁, η₂)...)
end

"""
    unconstrained_to_expectation(η₁::AbstractVector{<:Real}, η₂::AbstractMatrix{<:Real})

Convert from expecation parameters to mean and unconstrained upper-triangular matrix.
"""
function unconstrained_to_expectation(
    m::AbstractVector{<:Real},
    U::UpperTriangular{<:Real},
)
    return standard_to_expectation(unconstrained_to_standard(m, U)...)
end
