import Base.*
using TensorOperations

struct Mps{}
    matrices_                #
    contractions_
    contractionorder_
    function Mps(A::AbstractMatrix,B::AbstractMatrix...)
        M = [A,B...]
        Mps(M)
    end
    function Mps(M)
        #contractionorder
        order = optimalSequence1(size.(M))
        #contractions
        contractions = []
        for i in 1:length(M)                   # matrices' connections to the network
            push!(contractions, [-i, i])
        end
        push!(contractions, collect(1:length(M)))   # de tensors conectons
        # tensors
        tensors = Array{Array,1}(undef, length(M))
        tensors[1:end]=M

        return new(tensors, contractions, order)
    end
end

function optimalmul2(mps::Mps, T::AbstractArray)
    push!(mps.matrices_, T)
    res = ncon(mps.matrices_, mps.contractions_, order = mps.contractionorder_)
    pop!(mps.matrices_)
    return res
end



function optimalSequence1(array)
    #voor het algemene geval, brute force.
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




*(mps::Mps, T::AbstractArray) = optimalmul2(mps, T)
*(T::AbstractArray, mps::Mps) = optimalmul2(mps, T)

ten(res, A, K1, K2, K3) = @tensor res[i,j,k] = A[u,v,w]*K1[i,u]*K2[j,v]*K3[k,w]
*(A::Array{Float64,3}, K::Array{Array{Float64,2},1}) = ten(Array{Float64}(undef, size(A)), A, K[1], K[2],K[3])
*(K::Array{Array{Float64,2},1}, A::Array{Float64,3}) = ten(Array{Float64}(undef, size(A)), A, K[1], K[2],K[3])

#*(A::AbstractArray, K::Mps) = ten(Array{Float64}(undef, size(A)), A, K.matrices[1], K.matrices[2], K.matrices[3])
#*( K::Mps, A::AbstractArray) = ten(Array{Float64}(undef, size(A)), A, K.matrices[1], K.matrices[2], K.matrices[3])


"""

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
"""
