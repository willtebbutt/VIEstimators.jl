import Distributions: logpdf
import Random: rand
import Base: length
export Gaussian, normal_logpdf, GaussianPair, logpdf

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