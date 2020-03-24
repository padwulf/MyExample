include("MatrixProductState.jl")
include("SparseTensor.jl")

function SE(A::Array, kernels::Mps, activation, Y::SparseTensor)
    it = eachindex(Y)
    Yest = SparseTensor(activation.(kernels*A))
    it = eachindex(Y)
    d = (Yest[it] - Y[it])
    return sum(d .* d)
end
function SE_deriv(A::Array, kernels::Mps, activation, activation_deriv, Y::SparseTensor)
    z = kernels*A
    f = activation.(kernels*A)
    dldf = zeros(size(Y))
    for I in eachindex(Y)
        dldf[I] = 2*(f[I] - Y[I])
    end
    dldf_dfdz = dldf .* activation_deriv.(z)
    dldf_dfdz_dzda =  dldf_dfdz * kernels
    return dldf_dfdz_dzda
end

function L2(A::Array, kernels::Mps)
    return sum(A .* (A * kernels))
end
function L2_deriv(A::Array, kernels::Mps)
    return 2*A*kernels
end

self(x) = x
self_deriv(x) = 1
