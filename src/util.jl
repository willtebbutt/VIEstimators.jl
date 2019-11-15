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
