import Base: rand
export RPT∇, STL∇

# ReParametrisation Trick (RPT) gradient estimator. Assumes that `rand(q(ϕ))` is
# reparametrisable.
function RPT∇(rng::AbstractRNG, q, ϕ, log_π̃_over_q)
    elbo, back = Zygote.pullback(ϕ->log_π̃_over_q(ϕ, rand(rng, q(ϕ))), ϕ)
    ∂ϕ = first(back(1.0))
    return elbo, ∂ϕ
end

# Assumes single-sample. Multi-sample objectives should be viewed via AuxVI. No global
# parameters to learn for now. Include them later when this actually works.
function STL∇(rng::AbstractRNG, q, ϕ, log_π̃_over_q)

    # Compute approximation to elbo.
    z, back_z = Zygote.pullback(ϕ->rand(rng, q(ϕ)), ϕ)
    elbo, back_elbo = Zygote.pullback(log_π̃_over_q, ϕ, z)

    # Compute reverse-pass, dropping gradients appropriately.
    ∂ϕ_score, ∂z = back_elbo(1.0)
    ∂ϕ = first(back_z(∂z))

    return elbo, ∂ϕ, ∂ϕ_score
end
