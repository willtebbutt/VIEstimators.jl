export to_positive_definite, to_positive_definite_softplus, to_positive_definite_square,
    upper_triangular, to_positive_definite_softplus_inv


# Copy-Paste some util from Stheno.jl
diag_At_A(A::AbstractVecOrMat) = vec(sum(abs2.(A); dims=1))
diag_Xt_invA_X(A::Cholesky, X::AbstractVecOrMat) = diag_At_A(A.U' \ X)



"""
    to_positive_definite(U::UpperTriangular)

Exponentiate the diagonal of `U` so that `U'U` is positive definite.
"""
function to_positive_definite(U::UpperTriangular)
    new_data = copy(U.data)
    new_data[diagind(new_data)] .= exp.(diag(new_data)) .+ 1e-12
    return UpperTriangular(new_data)
end


# Custom adjoint necessary to handle mutation.
ZygoteRules.@adjoint function to_positive_definite(U::UpperTriangular)
    out = to_positive_definite(U)
    return out, function(Δ::AbstractMatrix)
        out_back = copy(Δ)
        out_back[diagind(out_back)] .= diag(Δ) .* diag(out)
        return (out_back,)
    end
end



"""
    to_positive_definite_softplus(U::UpperTriangular)

Exponentiate the diagonal of `U` so that `U'U` is positive definite.
"""
function to_positive_definite_softplus(U::UpperTriangular)
    new_data = copy(U.data)
    new_data[diagind(new_data)] .= softplus.(diag(new_data)) .+ 1e-12
    return UpperTriangular(new_data)
end

ZygoteRules.@adjoint function to_positive_definite_softplus(U::UpperTriangular)
    out = to_positive_definite_softplus(U)
    return out, function(Δ::AbstractMatrix)
        out_back = copy(Δ)
        out_back[diagind(out_back)] .= diag(Δ) .* logistic.(diag(U))
        return (out_back,)
    end
end



"""
    to_positive_definite_softplus_inv(U::UpperTriangular)

Inverse of to_positive_definite_softplus.
"""
function to_positive_definite_softplus_inv(U::UpperTriangular)
    new_data = copy(U.data)
    new_data[diagind(new_data)] .= invsoftplus.(diag(new_data) .- 1e-12)
    return UpperTriangular(new_data)
end

function frule(::typeof(to_positive_definite_softplus_inv), U::UpperTriangular)
    V = to_positive_definite_softplus_inv(U)
    function to_positive_definite_softplus_inv_pushforward(Δself, ΔU::AbstractMatrix)
        return to_positive_definite_softplus_inv_pushforward(Δself, UpperTriangular(ΔU))
    end
    function to_positive_definite_softplus_inv_pushforward(Δself, ΔU::UpperTriangular)
        new_data = copy(ΔU.data)
        new_data[diagind(new_data)] .= diag(new_data) .* exp.(diag(U) .- diag(V))
        return UpperTriangular(new_data)
    end
    return V, to_positive_definite_softplus_inv_pushforward
end


"""
    to_positive_definite_square(U::UpperTriangular)

Square the diagonal of `U` so that `U'U` is positive definite.
"""
function to_positive_definite_square(U::UpperTriangular)
    new_data = copy(U.data)
    new_data[diagind(new_data)] .= abs2.(diag(new_data)) .+ 1e-12
    return UpperTriangular(new_data)
end

ZygoteRules.@adjoint function to_positive_definite_square(U::UpperTriangular)
    out = to_positive_definite_square(U)
    return out, function(Δ::AbstractMatrix)
        out_back = copy(Δ)
        out_back[diagind(out_back)] .= 2 .* diag(Δ) .* diag(U)
        return (out_back,)
    end
end



"""
    upper_triangular(A::Matrix)

Exactly the same as the regular UpperTriangular function, except the adjoint returns a
dense matrix, rather than another UpperTriangular matrix. This is helpful when working with
(at least) the AMSGrad implementation in Flux.
"""
upper_triangular(A::Matrix) = UpperTriangular(A)

ZygoteRules.@adjoint function upper_triangular(A::Matrix)
    return upper_triangular(A), Δ->(collect(UpperTriangular(Δ)),)
end



#
# frule for cholesky
#

chol(A::Symmetric) = cholesky(A).U

function frule(::typeof(chol), A::Symmetric{T, Matrix{T}} where {T<:Real})
    U = chol(A)
    function chol_pushforward(Δself, ΔA::Symmetric)
        T = U' \ (U' \ ΔA')'
        T = UpperTriangular(T)
        T[diagind(T)] ./= 2
        return T * U
    end
    return U, chol_pushforward
end



#
# rrule for cholesky inv
#

ZygoteRules.@adjoint function inv(C::Cholesky{<:Real})
    Cinv = inv(C)
    return Cinv, function(Δ)
        return ((factors=-(C.U' \ (Δ + Δ')) * Cinv,),)
    end
end



# #
# # Helper functionality to make reverse-mode compute the correct thing...
# #

# chol_inv(S::Symmetric{<:Real}) = inv(cholesky(S))

# # Specialised rrule for natural_to_standard. This does things in a slightly strange way.
# ZygoteRules.@adjoint function chol_inv(S::Symmetric{<:Real})
#     Sinv = inv(cholesky(S))
#     return Sinv, Δ::AbstractMatrix{<:Real} -> (-Sinv'Δ * Sinv',)
# end



#
# cholesky specifically for Symmetric matrices.
#

# Implementation due to Seeger, Matthias, et al. "Auto-differentiating linear algebra."
ZygoteRules.@adjoint function cholesky(Σ::Symmetric{T, <:StridedMatrix{T}} where {T<:Real})
  C = cholesky(Σ)
  return C, function(Δ::NamedTuple)
    U, Ū = C.U, Δ.factors
    ΔΣ = Ū * U'
    ΔΣ = LinearAlgebra.copytri!(ΔΣ, 'U')
    ΔΣ = ldiv!(U, ΔΣ)
    BLAS.trsm!('R', 'U', 'T', 'N', one(eltype(Σ)), U.data, ΔΣ)
    ΔΣ ./= 2
    return (Symmetric(ΔΣ),)
  end
end
