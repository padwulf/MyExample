include("MatrixProductState.jl")
include("SparseTensor.jl")

using LinearAlgebra
import LinearAlgebra: eigen, \, det, logdet, inv


abstract type MultilinearKroneckerModel end


struct NSKRRegressor <: MultilinearKroneckerModel
    A_::AbstractArray
    λ_::Array{Float64}
    fill_::String
    function NSKRRegressor(size, λ, fill)
        A = 0.001*randn(size)
        new(A,λ,fill)
    end
end

function fit(model::NSKRRegressor, kernels::Array{T}, Y::SparseTensor) where T<:Eigen
    try
        Y = tensor(Y)
    catch e
        Y = tensor(Y, model.fill_)
    end
    @assert size(Y) == Tuple(length.([kernel.values for kernel in kernels]))
    G = Mps(Matrix.(G__.(kernels, model.λ_)))
    model.A_[:] = G*Y
end

function predict(model::NSKRRegressor, kernels::Array)
    K = Mps(kernels)
    return SparseTensor(model.A_*K)
end

function predict(model::NSKRRegressor, kernels::Array{T}, Y::SparseTensor) where T<:Eigen
    try
        Y = tensor(Y)
    catch e
        Y = tensor(Y, model.fill_)
    end
    @assert size(Y) == Tuple(length.([kernel.values for kernel in kernels]))
    H = Mps(Matrix.(H__.(kernels, model.λ_)))
    return SparseTensor(H*Y)
end

function predict_LOO(model::NSKRRegressor, kernels::Array{T}, Y::SparseTensor, setting::Tuple) where T<:Eigen
    try
        Y = tensor(Y)
    catch e
        Y = tensor(Y, model.fill_)
    end
    if setting==(0,)
        H = Mps(H__.(kernels, model.λ_))
        H_diag = Mps(Diagonal.(H.matrices))
        res = H*Y - H_diag*Y
        for I in CartesianIndices(res)
            div = 1
            for i in 1:length(I)
                div*= H.matrices[i][I[i], I[i]]
            end
            res[I] /= (1-div)
        end
    else
        hats = Matrix.(H__.(kernels, model.λ_))
        for i in setting
            hats[i] = H__LOO(hats[i])
        end
        H_LOO = Mps(hats)
        res = H_LOO*Y
    end
    return SparseTensor(res)
end



function G__(kernel::Eigen, λ::Float64)
    values = ones(length(kernel.values)) ./ (kernel.values + λ*ones(length(kernel.values)))
    return Eigen(values, kernel.vectors)
end
function H__(kernel::Eigen, λ::Float64)
    values = kernel.values ./ (kernel.values + λ*ones(length(kernel.values)))
    return Eigen(values, kernel.vectors)
end
function H__LOO(H)
    return (H - Diagonal(H)) ./ (ones(size(H)[1]) - diag(H))
end
