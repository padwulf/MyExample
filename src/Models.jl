#include("MatrixProductState.jl")
#include("SparseTensor.jl")
include("Attributes.jl")

using LinearAlgebra
using Optim
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

function predict(model::MultilinearKroneckerModel, kernels::Array)
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




struct KKRegressor <: MultilinearKroneckerModel
    A_::AbstractArray       #coefficients
    λ_::Array{Float64}      #hyperparam
    a_                      #activationfunction
    aderiv_
    L_                      #lossfunction
    Lderiv_
    R_                      #regularizationfunciton
    Rderiv_
    function KKRegressor(size, λ, a, L, R)
        if L==SE
            Lderiv = SE_deriv
        end
        if R == L2
            Rderiv = L2_deriv
        end
        if a == self
            aderiv = self_deriv
        end
        A = 0.001*randn(size)
        new(A,λ,a, aderiv,L,Lderiv,R,Rderiv)
    end
end

function fit(model::KKRegressor, kernels, Y::SparseTensor, init, optimizer)
    kernels = Mps(kernels)
    function objective(A, model, kernels, Y)
        #println(model.L_(A, kernels, model.a_, Y) + model.λ_[1]*model.R_(A, kernels))
        r =  model.L_(A, kernels, model.a_, Y)  + model.λ_[1]*model.R_(A, kernels)
        println(r)
        return r
    end
    a = objective(model.A_, model, kernels, Y)
    function objective_deriv(G, A, model, kernels, Y)
        r = model.Lderiv_(A, kernels, model.a_, model.aderiv_, Y) + model.λ_[1]*model.Rderiv_(A,kernels)
        G[:]  = r[:]
        return G
    end
    b = objective_deriv(randn(size(Y)),model.A_, model, kernels, Y)
    b = optimize(A -> objective(A, model, kernels, Y), (G,A) -> objective_deriv(G, A, model, kernels, Y), init, optimizer, Optim.Options(iterations=100))
    model.A_[:]=b.minimizer[:]
    @assert model.A_[:]==b.minimizer[:]
    return b

end
function fit(model::KKRegressor, kernels, Y::SparseTensor, optimizer)
    return fit(model, kernels, Y, model.A_, optimizer)
end
function fit(model::KKRegressor, kernels, Y::SparseTensor)
    return fit(model, kernels, Y, model.A_, LBFGS())
end
