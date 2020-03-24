include("Models.jl")
include("Scores.jl")


import Random.randperm
import Random.shuffle!
using Statistics
using Optim

function splitn(x::AbstractArray, n::Int)
    shuffle!(x)
    s = length(x) / n
    [x[round(Int64, (i-1)*s)+1:min(length(x),round(Int64, i*s))] for i=1:n]
end
function Kfold(inds::Array, setting::Tuple, nfolds)
    if setting == (0,)
        s = splitn(inds, nfolds)
    elseif true # setting where all indices omitted should be implemented differently
        maxi = maximum(inds)
        tofreelyselect=setdiff(collect(1:length(maxi)), setting)
        d = Dict()
        for I in inds
            key = [I[i] for i in tofreelyselect]
            if key in keys(d)
                push!(d[key], I)
            else
                d[key]=[I]
            end
        end
        s_keys = splitn(collect(keys(d)), nfolds)
        s = [[] for i in 1:length(s_keys)]
        for i in 1:length(s_keys)
            for j in 1:length(s_keys[i])
                s[i] = cat(s[i], d[s_keys[i][j]], dims=1)
            end
        end

    else
        println("other settings not yet implemented")
    end
    folds = []
    for i in 1:nfolds
        a = (train, test) = vcat(s[1:end .!= i]...), s[i]
        push!(folds,a)
    end
    return folds
end

function CVestimate(model::MultilinearKroneckerModel, kernels, Y::SparseTensor, folding, setting::Tuple)
    #determine the folds
    if folding == "LOO"
        return predict_LOO(model, kernels, Y, setting)
    elseif typeof(folding) <: Int
        folds = Kfold(eachindex(Y), setting, folding)
    elseif typeof(folding) <: AbstractArray
        folds = folding
    else
        throw("Input for nfolds not valid")
    end
    Ycvest = SparseTensor(size(Y))
    for fold in folds
        traininds, testinds = fold
        Y_train = SparseTensor(Y, traininds)
        model.A_[:]=0.001*randn(size(model.A_))[:]
        fit(model, kernels, Y_train)
        Yest = predict(model, kernels)
        for I in testinds
            Ycvest[I] = Yest[I]
        end
    end
    return Ycvest
end

function CVscore(model::MultilinearKroneckerModel, kernels, Y::SparseTensor, folding, setting::Tuple, score)
    Ycvest = CVestimate(model, kernels, Y, folding, setting)
    @assert isempty(setdiff(eachindex(Y), eachindex(Ycvest)))
    inds = eachindex(Y)
    Y = Y[inds]
    Ycvest = Ycvest[inds]
    return score(Ycvest, Y)
end

function CVscore(model::MultilinearKroneckerModel, kernels, Y::SparseTensor, folding, setting::Tuple, score, times)
    scores = []
    for i in 1:times
        sc = CVscore(model, kernels, Y, folding, setting, score)
        push!(scores, sc)
    end
    return mean(scores),3*std(scores)
end
# when all hyperparameters the same:
function optimizeHyperParameters(model::MultilinearKroneckerModel, kernels, Y::SparseTensor, folding, setting::Tuple, score)
    model = deepcopy(model)
    function objective(λ, model::MultilinearKroneckerModel, kernels, Y::SparseTensor, folding, setting::Tuple, score)
        model.λ_[:]=λ[:]
        println(model.λ_)
        return -CVscore(model, kernels, Y, folding, setting, score)
    end
    b = optimize(λ -> objective([λ for i in 1:length(model.λ_)],model, kernels,Y,folding,setting,score), 0.001, 1000)
    minimizer = [b.minimizer for i in 1:length(model.λ_)]
    model.λ_[:]=minimizer[:]
    return model, b
end
#when differetn hyperparemeters
function optimizeHyperParameters(model::MultilinearKroneckerModel, kernels, Y::SparseTensor, folding, setting::Tuple, score, optimizer, init)
    model = deepcopy(model)
    function objective(λ, model::MultilinearKroneckerModel, kernels, Y::SparseTensor, folding, setting::Tuple, score)
        model.λ_[:]=λ[:]
        println(model.λ_)
        return -CVscore(model, kernels, Y, folding, setting, score)
    end
    lower = zeros(length(model.λ_))
    upper = [Inf for i in 1:length(model.λ_)]
    inner_optimizer = optimizer
    b = optimize(λ -> objective(λ,model, kernels,Y,folding,setting,score), lower, upper, init, Fminbox(inner_optimizer))
    minimizer = b.minimizer
    model.λ_[:]=minimizer[:]
    return model, b
end
