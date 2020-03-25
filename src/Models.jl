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

function fit(model::NSKRRegressor, kernels, Y::SparseTensor)
    println("Diagonalizing kernels..")
    kernelseigen = eigen.(kernels)
    println("Diagonalizing kernels done.")
    fit(model, kernelseigen, Y)
end
function fit(model::NSKRRegressor, kernels::Array{T}, Y::SparseTensor) where T<:Eigen
    if !full(Y)
        if model.fill_=="No fill"
            fit_iteratively(model::NSKRRegressor, kernels::Array{T}, Y::SparseTensor)
        else
            Y = tensor(Y, model.fill_)
            @assert size(Y) == Tuple(length.([kernel.values for kernel in kernels]))
            G = Mps(Matrix.(G__.(kernels, model.λ_)))
            model.A_[:] = G*Y
        end
    else
        Y = tensor(Y)
        @assert size(Y) == Tuple(length.([kernel.values for kernel in kernels]))
        G = Mps(Matrix.(G__.(kernels, model.λ_)))
        model.A_[:] = G*Y
    end
end
function fit_iteratively(model::NSKRRegressor, kernels::Array{T}, Y::SparseTensor) where T<:Eigen
    return fit_iteratively(model, kernels, Y, model.A_, LBFGS())
end
function fit_iteratively(model::NSKRRegressor, kernels::Array{T}, Y::SparseTensor, init, optimizer) where T<:Eigen
    inversehats = Mps(Matrix.(Hinverse__.(kernels, model.λ_)))
    kernels = Mps(Matrix.(kernels))
    b = optimize(A -> Loss_NSKR(A,Y,kernels, inversehats), (G,A)->Loss_NSKR_deriv(G,A,Y,kernels, inversehats), init, optimizer)
    model.A_[:] = b.minimizer
    return b
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

function predict_LOO(model::NSKRRegressor, kernels, Y::SparseTensor, setting::Tuple)
    println("Diagonalizing kernels..")
    kernelseigen = eigen.(kernels)
    println("Diagonalizing kernels done.")
    predict_LOO(model, kernelseigen, Y,setting)
end
function predict_LOO(model::NSKRRegressor, kernels::Array{T}, Y::SparseTensor, setting::Tuple) where T<:Eigen
    try
        Y = tensor(Y)
    catch e
        Y = tensor(Y, model.fill_)
    end
    if setting==Tuple([false for i in 1:length(setting)])
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
        for i in 1:length(setting)
            if setting[i]
                hats[i] = H__LOO(hats[i])
            end
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
function Hinverse__(kernel::Eigen, λ::Float64)
    values = (kernel.values + λ*ones(length(kernel.values))) ./  kernel.values
    return Eigen(values, kernel.vectors)
end
function H__LOO(H)
    return (H - Diagonal(H)) ./ (ones(size(H)[1]) - diag(H))
end
function Loss_NSKR(A::AbstractArray, Y::AbstractArray, kernels::Mps, inversehats:: Mps)
    Yest = SparseTensor(kernels*A)
    ridgepart = inversehats * Yest - Yest
    inds = eachindex(Y)
    Yinds = Y[inds]
    Yestinds = Yest[inds]
    l_ = sum((Yinds-Yestinds) .* (Yinds-Yestinds)) + sum( Yest .* ridgepart)
    println(l_)
    return l_
end
function Loss_NSKR_deriv(G, A::AbstractArray, Y::AbstractArray, kernels::Mps, inversehats:: Mps)
    Yest = kernels * A
    dldf = zeros(size(Y))
    for I in eachindex(Y)
        dldf[I] = Yest[I] - Y[I]
    end
    dldf += ( inversehats *Yest - Yest)
    dldfdfdA = kernels * dldf
    G[:,:,:] = dldfdfdA
    #println(("deriv ",sum(G.*G)))
    return G
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
        r =  model.L_(A, kernels, model.a_, Y)  + model.λ_[1]*model.R_(A, kernels)
        #println(r)
        return r
    end
    function objective_deriv(G, A, model, kernels, Y)
        r = model.Lderiv_(A, kernels, model.a_, model.aderiv_, Y) + model.λ_[1]*model.Rderiv_(A,kernels)
        G[:]  = r[:]
        return G
    end
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
function predict(model::KKRegressor, kernels)
    K = Mps(kernels)
    return SparseTensor(model.a_(K*model.A_))
end
