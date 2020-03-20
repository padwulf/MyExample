import Base.*
using TensorOperations

# verbetering: kijken of MPS ook Eigen kan bevatten en daarmee ncon doen. Indien niet kan mss nuttig zijn
# zelf een implementatie voor tensor*matrix. met matrix een Eigen. Dit kan mss veel sneller.

struct Mps{}
    matrices::Array{AbstractMatrix}
    optimalOrder::Vector{Int}
    function Mps(A::AbstractMatrix,B::AbstractMatrix...)
        return new([A,B...], zeros(length([A,B...])))
    end
    function Mps(M)
        return new(M, zeros(length(M)))
    end
end

function optimalSequence1(array)
    product = prod(last.(array))
    array = [[array[i][1], array[i][2], i] for i in 1:length(array)]       #third element is indexnumber
    (optcost, optseq) =  costPermutatingrecursive([], array, product, 0, Inf, array)
    return [t[3] for t in optseq]                                                   #return only indexnumbers of permutation
end
setOptimalOrder(mps::Mps, o::Vector{Int}) = (mps.optimalOrder[1:end]=o)


function costPermutatingrecursive(curlistperm, array, product, cost, optimalcost, optimalsequence)
    if length(array)==1
        push!(curlistperm, array[1])
        cost += product*array[1][1]
        r_optimalcost = optimalcost
        r_optimalsequence = optimalsequence
        if cost<optimalcost
            r_optimalsequence = deepcopy(curlistperm)
            r_optimalcost = cost
        end
        pop!(curlistperm)
        return (r_optimalcost, r_optimalsequence)
    else
        for i in 1:length(array)
            #adapt for current brach in seach tree
            newcurlistperm = push!(curlistperm, array[i])
            newarray = array[1:end .!= i]
            cost += product*array[i][1]
            product *= array[i][1]/array[i][2]
            (optimalcost, optimalsequence) = costPermutatingrecursive(newcurlistperm, newarray, product, cost, optimalcost, optimalsequence)
            #reconstruct for descending other branches in de search tree (in place dept first)
            #! first reconstruct product again since used in reconstruction of cost
            pop!(curlistperm)
            product *= array[i][2]/array[i][1]
            cost -= product*array[i][1]
        end
    return (optimalcost, optimalsequence)
    end
end

function optimalmul(mps::Mps, T::AbstractArray)
    @assert last.(size.(mps.matrices)) == [s for s in size(T)]

    ##sequence:
    if mps.optimalOrder[1] == 0
        seqq = optimalSequence1(size.(mps.matrices))
        setOptimalOrder(mps, seqq)
    end

    ##network
    network = []
    for i in 1:length(mps.matrices)                   # matrices' connections to the network
        push!(network, [-i, i])
    end
    push!(network, collect(1:length(mps.matrices)))     # tensor connections to the network

    ##tensors
    tensors = Array{Array,1}(undef, length(mps.matrices)+1)
    tensors[end]=T
    tensors[1:end-1]=mps.matrices
    ncon(tensors, network, order=mps.optimalOrder)
end


*(mps::Mps, T::AbstractArray) = optimalmul(mps, T)
*(T::AbstractArray, mps::Mps) = optimalmul(mps, T)
