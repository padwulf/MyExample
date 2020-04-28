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
    n_iter_::Int
    function NSKRRegressor(size, λ, fill::String)
        A = 0.001*randn(size)
        new(A,λ,fill,0)
    end
    function NSKRRegressor(size, λ, n_iter::Int)
        A = 0.001*randn(size)
        new(A,λ,"No fill", n_iter)
    end
end
function fit(model::NSKRRegressor, K, Y, inds)
    println("Diagonalizing kernels..")
    Ke = eigen.(K)
    println("Diagonalizing kernels done.")
    fit(model, Ke, Y,inds)
end
function fit(model::NSKRRegressor, K::Array{T}, Y, inds) where T<:Eigen
    if length(inds) < length(Y)             #if not all indices in Y are observed
        if model.fill_=="No fill"                   # if not filled: fit_iteratively
            fit_iteratively(model::NSKRRegressor, K::Array{T}, Y, inds)
        else                                        # else: fill according to model.fill_
            Yfilled = fill(Y, inds, model.fill_)
            G = Mps(Matrix.(G__.(K, model.λ_)))
            model.A_[:] = G*Yfilled
        end
    else                                    #else: completely observed; calculate directly
        G = Mps(Matrix.(G__.(K, model.λ_)))
        model.A_[:] = G*Y
    end
end
function fit_iteratively(model::NSKRRegressor, K::Array{T}, Y, inds) where T<:Eigen
    return fit_iteratively(model, K, Y, inds, 0.001*randn(size(Y)), LBFGS())
end
function fit_iteratively(model::NSKRRegressor, K::Array{T}, Y, inds, init, optimizer) where T<:Eigen
    inversehats = Mps(Matrix.(Hinverse__.(K, model.λ_)))
    K = Mps(Matrix.(K))
    @time opt = optimize(Optim.only_fg!((F,G,A) -> NSKRR_loss!(F,G,A,K,inversehats, Y, inds)), init, optimizer, Optim.Options(store_trace=true, iterations = model.n_iter_))
    model.A_[:] = opt.minimizer
    return opt
end

function predict(model::NSKRRegressor, K::Array)
    K = Mps(K)
    return model.A_*K
end

function predict(model::NSKRRegressor, K::Array{T}) where T<: Eigen
    K = Mps(Matrix.(K))
    return model.A_*K
end

function predict(model::NSKRRegressor, kernels::Array{T}, Y) where T<:Eigen
    #Yfilled = fill(Y, inds, model.fill_)
    Yfilled = Y
    H = Mps(Matrix.(H__.(kernels, model.λ_)))
    return H*Yfilled
end

function predict_LOO(model::NSKRRegressor, kernels, Y, inds, setting::Tuple)
    println("Diagonalizing kernels..")
    kernelseigen = eigen.(kernels)
    println("Diagonalizing kernels done.")
    predict_LOO(model, kernelseigen, Y, inds,setting)
end

function predict_LOO(model::NSKRRegressor, kernels::Array{T}, Y, inds, setting::Tuple) where T<:Eigen
    Yfilled = fill(Y, inds, model.fill_)
    if setting==Tuple([false for i in 1:length(setting)])
        H = Mps(Matrix.(H__.(kernels, model.λ_)))
        H_diag = Mps(Diagonal.(H.matrices_))
        res = H*Yfilled - H_diag*Yfilled
        for I in CartesianIndices(res)
            div = 1
            for i in 1:length(I)
                div*= H.matrices_[i][I[i], I[i]]
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
        res = H_LOO*Yfilled
    end
    return res
end

#muliplyer to estimate interaction coefficients A_ from labels Y_
function G__(kernel::Eigen, λ::Float64)
    values = ones(length(kernel.values)) ./ (kernel.values + λ*ones(length(kernel.values)))
    return Eigen(values, kernel.vectors)
end
#muliplyer to estimate predicted values from labels Y
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



struct KKRegressor <: MultilinearKroneckerModel
    A_::AbstractArray
    λ_::Array{Float64}
    n_iter::Int
    a_                      #activationfunction
    aderiv_
    L_                      #lossfucniton
    Lderiv_
    R_
    Rderiv_

    function KKRegressor(size, λ, n_iter, type)
        if type == "Regression"
            a,aderiv = self,self_deriv
            L,Lderiv = se, se_deriv
            R,Rderiv = l2, l2_deriv
        elseif type == "Classification"
            a,aderiv = sigmoid,sigmoid_deriv
            L,Lderiv = bce, bce_deriv
            R,Rderiv = l2, l2_deriv
        else
            println("not implemented")
        end
        new(Array{Float64}(undef, size),λ, n_iter, a, aderiv, L, Lderiv, R, Rderiv)
    end
end

function fit(model::KKRegressor, K, Y, inds)
    A0 = 0.001*randn(size(Y))
    λ = model.λ_[1]
    iterations = model.n_iter
    @time opt = optimize(Optim.only_fg!((F,G,A) -> Loss_reg!(F,G,A,K,model.a_, model.aderiv_, λ, Y,inds, model.L_, model.Lderiv_, model.R_, model.Rderiv_)), A0, LBFGS(), Optim.Options(store_trace=true, iterations = iterations))

    """
    if model.a_== self && model.L_ == SE_reg!
        @time opt = optimize(Optim.only_fg!((F,G,A) -> SE_reg!(F,G,A,K,λ,Y,inds)), A0, LBFGS(), Optim.Options(store_trace=true, iterations = iterations))
        #@time opt = optimize(Optim.only_fg!((F,G,A) -> Loss_reg!(F,G,A,K,self, self_deriv, λ, Y,inds, se, se_deriv, l2, l2_deriv)), A0, LBFGS(), Optim.Options(store_trace=true, iterations = iterations))

        #F,G,A,K,act,actderiv,λ,Y,inds, loss,lossderiv, reg, regderiv
    elseif model.L_ == BCE_reg!
        @time opt = optimize(Optim.only_fg!((F,G,A) -> BCE_reg!(F,G,A,K, model.a_,model.aderiv_ ,λ,Y,inds)), A0, LBFGS(), Optim.Options(store_trace=true, iterations = iterations))
    else
        println("other")
    end
    """
    model.A_[:] = opt.minimizer
    return opt
end
function predict(model::KKRegressor, K)
    return model.a_.(model.A_*K)
end

struct KKTRegressor <: MultilinearKroneckerModel
    rank_::Int
    D_::AbstractArray
    U_::AbstractArray
    λ_::Array{Float64}
    n_iter::Int
    a_                      #activationfunction
    aderiv_
    L_                      #lossfucniton
    Lderiv_
    R_
    Rderiv_

    function KKTRegressor(size, λ, n_iter, type, rank)
        if type == "Regression"
            a,aderiv = self,self_deriv
            L,Lderiv = se, se_deriv
            R,Rderiv = l2, l2_deriv
        elseif type == "Classification"
            a,aderiv = sigmoid,sigmoid_deriv
            L,Lderiv = bce, bce_deriv
            R,Rderiv = l2, l2_deriv
        else
            println("not implemented")
        end
        U = [0.1*randn(size[i], rank) for i in 1:length(size)]
        D = 0.1*randn(rank)
        new(rank, D, U,λ, n_iter, a, aderiv, L, Lderiv, R, Rderiv)
        LossTucker_reg!(0,0,D,U, K,0,0,0,0,inds,0,0,0,0)
    end
end

# For the tucker based model with D and U's:
# Loss: generic for all combinations in losses and regs: F and G are value and derivative w.r.t. D, U's
# l = given as argument = l(y,f)  with f = a(A*K)
# r = given as argumnet = r(A,K)
# note that Y is a dense label tensor. inds denote what labels are really observed to evaluate the lossfunction.
function LossTucker_reg!(F,G,D,U,K,act,actderiv,λ,Y,inds, loss,lossderiv, reg, regderiv)
    KU = K .* U
    z = zeros(length(inds))
    print(size(z))
    for i in 1:length(z)
        I = inds[i]
        res = D
        for j in 1:length(I)
            res = res .* KU[j][I[j],:]
        end
        print(res)
        z[i] = sum(res)
    end
    return(z)
end
"""
inds
KKTRegressor(size(Y), [0.1], 50, "Regression", 3)
U
using ITensors

i,j,k = Index(15), Index(15), Index(15)
A = ITensor(i,j,k)
A[i=>1,j=>1,k=>1] = 11.1
B = randomITensor(i,j,k)
A[i(1), j(2), k(10)] = 5.0

A[i(1), :, :]

K = randomITensor(i,j[2:5])

c = CartesianIndices(A)
A[1,1,1]

A*B

SP = diagITensor(i,j,k)
SP[1,2,1] = 5


Yinds = K1inds = i,j,k = Index(15, "i"), Index(15, "j"), Index(15, "k")
U1inds = K2inds = u,v,w = Index(15, "u"), Index(15, "v"), Index(15, "w")
rinds = U2inds = r1,r2,r3 = Index(5, "r1"), Index(5, "r2"), Index(5, "r3")
r = Index(5, "r")

D = diagITensor(r,r,r)
U1, U2, U3 = randomITensor(r,u), randomITensor(r,v), randomITensor(r,w)
K1, K2, K3 = randomITensor(i,u), randomITensor(j,v), randomITensor(k,w)

K1*U1*K2

D*U1*U2*U3

U1*U2*U3*K1*K2*K3

pre = "/home/padwulf/Documents/Experiments/Datasets/Artificial/Classification/0.5_(15, 15, 15)_(1, 1, 1)"
Y, inds = readtensor(pre*"/data.txt")
K1 = readdlm(pre*"/k1.txt")
K2 = readdlm(pre*"/k2.txt")
K3 = readdlm(pre*"/k3.txt")
K = [K1,K2,K3]
Ke = eigen.(K)
"""
