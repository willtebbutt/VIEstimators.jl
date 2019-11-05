import Distributions: logpdf
import Random: rand
import Base: length
export Gaussian, normal_logpdf, StandardGaussian, GaussianPair, logpdf
export standard_to_natural, natural_to_standard
export standard_to_expectation, expectation_to_standard

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



"""
    natural_to_standard(θ₁::AbstractVector{<:Real}, θ₂::AbstractMatrix{<:Real})

Convert from natural to standard parametrisation.
"""
function natural_to_standard(θ₁::AbstractVector{<:Real}, θ₂::AbstractMatrix{<:Real})
    C = cholesky((-2) .* θ₂)
    S = inv(C)
    return S * θ₁, S
end

"""
    standard_to_natural(m::AbstractVector{<:Real}, S::AbstractMatrix{<:Real})

Convert from standard to natural parametrisation.
"""
function standard_to_natural(m::AbstractVector{<:Real}, S::AbstractMatrix{<:Real})
    C = cholesky(S)
    return C \ m, inv(C) ./ (-2)
end

"""
    expectation_to_standard(η₁::AbstractVector{<:Real}, η₂::AbstractMatrix{<:Real})

Convert from expectation to standard parametrisation.
"""
function expectation_to_standard(η₁::AbstractVector{<:Real}, η₂::AbstractMatrix{<:Real})
    return η₁, η₂ - η₁ * η₁'
end

"""
    standard_to_natural(m::AbstractVector{<:Real}, S::AbstractMatrix{<:Real})

Convert from standard to natural parametrisation.
"""
function standard_to_expectation(m::AbstractVector{<:Real}, S::AbstractMatrix{<:Real})
    return m, S + m * m'
end
