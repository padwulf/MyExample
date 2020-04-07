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

function Kfold(cartesianinds::Array, setting::Tuple, nfolds)
    if setting == Tuple([false for i in 1:length(setting)]) #setting A
        buckets = splitn(cartesianinds, nfolds)
    elseif setting == Tuple([true for i in 1:length(setting)])
        throw("kfolding for this setting not yet implemented")
    else
        d = Dict() # keys: contain indices where to freely select, values: all observed CartesianIndices according to a key
        for I in cartesianinds
            key = [I[i] for i in 1:length(setting) if setting[i]==false]
            if key in keys(d)
                push!(d[key], I)
            else
                d[key]=[I]
            end
        end
        buckets_keys = splitn(collect(keys(d)), nfolds)
        buckets = [[] for i in 1:length(buckets_keys)]
        for i in 1:length(buckets)
            for j in 1:length(buckets_keys[i])
                buckets[i] = cat(buckets[i], d[buckets_keys[i][j]], dims=1)
            end
        end
    end
    folds = []
    for i in 1:nfolds
        a = (train, test) = vcat(buckets[1:end .!= i]...), buckets[i]
        push!(folds,a)
    end
    return folds

end
function CVestimate(model::MultilinearKroneckerModel, kernels, Y, inds, folding, setting::Tuple)
    #determine the folds
    if folding == "LOO"
        return predict_LOO(model, kernels, Y, inds, setting)
    elseif typeof(folding) <: Int
        folds = Kfold(inds, setting, folding)
    elseif typeof(folding) <: AbstractArray
        folds = folding
    else
        throw("Input for nfolds not valid")
    end
    Ycvest = Array{Float64}(undef, size(Y))
    for fold in folds
        traininds, testinds = fold
        model.A_[:]=0.001*randn(size(model.A_))[:]
        fit(model, kernels, Y, traininds)
        Yest = predict(model, kernels)
        for I in testinds
            Ycvest[I] = Yest[I]
        end
    end
    return Ycvest
end

function CVscore(model::MultilinearKroneckerModel, kernels, Y, inds, folding, setting::Tuple, score)
    Ycvest = CVestimate(model, kernels, Y, inds, folding, setting)
    Y = [Y[inds[i]] for i in 1:length(inds)]
    Ycvest = [Ycvest[inds[i]] for i in 1:length(inds)]
    return score(Ycvest, Y)
end

function CVscore(model::MultilinearKroneckerModel, kernels, Y, inds, folding, setting::Tuple, score, times)
    scores = []
    for i in 1:times
        sc = CVscore(model, kernels, Y, inds, folding, setting, score)
        push!(scores, sc)
    end
    return mean(scores),3*std(scores)
end

# when all hyperparameters the same:
function optimizeHyperParameters(model::MultilinearKroneckerModel, kernels, Y, inds, folding, setting::Tuple, score)
    lower = 0.001
    upper = 10000
    optimizer = Brent()
    optimizeHyperParameters(model, kernels, Y, inds, folding, setting, score, lower, upper, optimizer)
end
function optimizeHyperParameters(model::MultilinearKroneckerModel, kernels, Y, inds, folding, setting::Tuple, score, lower, upper, optimizer)
    function objective(λ, model::MultilinearKroneckerModel, kernels, Y, inds, folding, setting::Tuple, score)
        model.λ_[:]=λ[:]
        return -CVscore(model, kernels, Y, inds, folding, setting, score)
    end
    opt = optimize(λ -> objective([λ for i in 1:length(model.λ_)],model, kernels,Y, inds,folding,setting,score), lower, upper, optimizer, store_trace=true)
    model.λ_[:]=[opt.minimizer for i in 1:length(model.λ_)]
    return opt
end

#when differetn hyperparemeters
function optimizeHyperParameters(model::MultilinearKroneckerModel, kernels, Y, inds, folding, setting::Tuple, score, optimizer, init)
    function objective(λ, model::MultilinearKroneckerModel, kernels, Y, inds, folding, setting::Tuple, score)
        model.λ_[:]=λ[:]
        #println(model.λ_)
        return -CVscore(model, kernels, Y, inds, folding, setting, score)
    end
    lower = zeros(length(model.λ_))
    upper = [Inf for i in 1:length(model.λ_)]
    inner_optimizer = optimizer
    opt = optimize(λ -> objective(λ,model, kernels,Y, inds,folding,setting,score), lower, upper, init, Fminbox(inner_optimizer), Optim.Options(store_trace=true))
    model.λ_[:]=opt.minimizer[:]
    return opt
end
