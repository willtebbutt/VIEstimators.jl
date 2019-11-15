using FiniteDifferences, ChainRulesCore

# Make FiniteDifferences work with some of the types in this package. Shame this isn't
# automated...

import FiniteDifferences: to_vec

# Make FiniteDifferences work for any user-defined type that has fields. This assumes that
# the type passed in admits a constructor that just takes all of the fields and produces a
# new object. As such, this won't work in the general case, but should work for all of the
# stuff in this package.
function to_vec(x::T) where {T}
    isempty(fieldnames(T)) && throw(error("Expected some fields. None found."))
    vecs_and_backs = map(name->to_vec(getfield(x, name)), fieldnames(T))
    vecs, backs = first.(vecs_and_backs), last.(vecs_and_backs)
    x_vec, back = to_vec(vecs)
    return x_vec, function(x′_vec)
        vecs′ = back(x′_vec)
        return T(map((back, vec)->back(vec), backs, vecs′)...)
    end
end


# My version of isapprox
function fd_isapprox(x_ad::Nothing, x_fd, rtol, atol)
    return fd_isapprox(x_fd, zero(x_fd), rtol, atol)
end
function fd_isapprox(x_ad::AbstractArray, x_fd::AbstractArray, rtol, atol)
    return all(fd_isapprox.(x_ad, x_fd, rtol, atol))
end
function fd_isapprox(x_ad::Real, x_fd::Real, rtol, atol)
    return isapprox(x_ad, x_fd; rtol=rtol, atol=atol)
end
function fd_isapprox(x_ad::NamedTuple, x_fd, rtol, atol)
    f = (x_ad, x_fd)->fd_isapprox(x_ad, x_fd, rtol, atol)
    return all([f(getfield(x_ad, key), getfield(x_fd, key)) for key in keys(x_ad)])
end
function fd_isapprox(x_ad::Tuple, x_fd::Tuple, rtol, atol)
    return all(map((x, x′)->fd_isapprox(x, x′, rtol, atol), x_ad, x_fd))
end
function fd_isapprox(x_ad::Dict, x_fd::Dict, rtol, atol)
    return all([fd_isapprox(get(()->nothing, x_ad, key), x_fd[key], rtol, atol) for
        key in keys(x_fd)])
end


function adjoint_test(
    f, ȳ, x...;
    rtol=1e-9,
    atol=1e-9,
    fdm=FiniteDifferences.Central(5, 1),
    print_results=false,
    test=true,
)

    # Compute forwards-pass and j′vp.
    y, back = Zygote.forward(f, x...)
    adj_ad = back(ȳ)
    adj_fd = j′vp(fdm, f, ȳ, x...)

    # If unary, pull out first thing from ad.
    adj_ad = length(x) == 1 ? first(adj_ad) : adj_ad

    # Check that forwards-pass agrees with plain forwards-pass.
    test && @test y ≈ f(x...)

    # Check that ad and fd adjoints (approximately) agree.
    print_results && print_adjoints(adj_ad, adj_fd, rtol, atol)
    test && @test fd_isapprox(adj_ad, adj_fd, rtol, atol)

    return adj_ad, adj_fd
end

function print_adjoints(adjoint_ad, adjoint_fd, rtol, atol)
    @show typeof(adjoint_ad), typeof(adjoint_fd)
    adjoint_ad, adjoint_fd = to_vec(adjoint_ad)[1], to_vec(adjoint_fd)[1]
    println("atol is $atol, rtol is $rtol")
    println("ad, fd, abs, rel")
    abs_err = abs.(adjoint_ad .- adjoint_fd)
    rel_err = abs_err ./ adjoint_ad
    display([adjoint_ad adjoint_fd abs_err rel_err])
    println()
end


const _fdm = central_fdm(5, 1)

function ensure_not_running_on_functor(f, name)
    # if x itself is a Type, then it is a constructor, thus not a functor.
    # This also catchs UnionAll constructors which have a `:var` and `:body` fields
    f isa Type && return

    if fieldcount(typeof(f)) > 0
        throw(ArgumentError(
            "$name cannot be used on closures/functors (such as $f)"
        ))
    end
end

"""
    frule_test(f, (x, ẋ)...; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)
# Arguments
- `f`: Function for which the `frule` should be tested.
- `x`: input at which to evaluate `f` (should generally be set to an arbitary point in the domain).
- `ẋ`: differential w.r.t. `x` (should generally be set randomly).
All keyword arguments except for `fdm` are passed to `isapprox`.
"""
function frule_test(f, (x, ẋ); rtol=1e-9, atol=1e-9, fdm=_fdm, kwargs...)
    return frule_test(f, ((x, ẋ),); rtol=rtol, atol=atol, fdm=fdm, kwargs...)
end

function frule_test(f, xẋs::Tuple{Any, Any}...; rtol=1e-9, atol=1e-9, fdm=_fdm, kwargs...)
    ensure_not_running_on_functor(f, "frule_test")
    xs, ẋs = collect(zip(xẋs...))
    Ω, pushforward = ChainRulesCore.frule(f, xs...)
    @test f(xs...) == Ω
    dΩ_ad = pushforward(NamedTuple(), ẋs...)

    # Correctness testing via finite differencing.
    dΩ_fd = jvp(fdm, xs->f(xs...), (xs, ẋs))
    @test isapprox(
        collect(dΩ_ad),  # Use collect so can use vector equality
        collect(dΩ_fd);
        rtol=rtol,
        atol=atol,
        kwargs...
    )
end
