using DelimitedFiles
import Base.size
import Base.getindex
import Base.setindex!
import Base.eachindex
import Base.CartesianIndices

struct SparseTensor{T,N} <: AbstractArray{T,N}
    # T is set to Float64. Should be ...
    elements_::Dict
    size_
    function SparseTensor(filename::String)
        elements = Dict()
        a = readdlm(filename, Float64)
        for i in 1:size(a)[1]
            index = CartesianIndex(Tuple(convert(Array{Int}, a[i,1:end-1])))
            elements[index] = a[i,end]
        end
        @assert length(keys(elements)) == length(unique(keys(elements)))
        s = Tuple(maximum(keys(elements)))
        new{Float64, length(a[1,:])-1}(elements, s)
    end
    function SparseTensor(elements::Dict)
        s = Tuple(maximum(keys(T.elements_)))
        new{Float64, length(collect(keys(elements))[1])}(elements, s)
    end
    function SparseTensor(elements::Dict, s::Tuple)
        new{Float64, length(collect(keys(elements))[1])}(elements, s)
    end
    function SparseTensor(s::Tuple)
        new{Float64, length(s)}(Dict(), s)
    end
end
function SparseTensor(old::SparseTensor, selection::Array)
    new = SparseTensor(size(old))
    for I in selection
        new[I] = old[I]
    end
    return new
end
function SparseTensor(a::Array)
    d = Dict()
    for I in CartesianIndices(a)
        d[I] = a[I]
    end
    return SparseTensor(d, size(a))
end

function size(T::SparseTensor)
    return T.size_
end
function getindex(T::SparseTensor, ind::CartesianIndex)
    for i in 1:length(ind)
        if ind[i]>T.size_[i]
            throw("index out of range")
        end
    end
    try
        return T.elements_[ind]
    catch e
        return missing
    end
end
function getindex(T::SparseTensor, ind...)
    ind = CartesianIndex(ind)
    for i in 1:length(ind)
        if ind[i]>T.size_[i]
            throw("index out of range")
        end
    end
    try
        return T.elements_[ind]
    catch e
        return missing
    end
end
function setindex!(T::SparseTensor, v, ind::CartesianIndex)
    for i in 1:length(ind)
        if ind[i]>T.size_[i]
            throw("index out of range")
        end
    end
    T.elements_[ind] = v
end
function setindex!(T::SparseTensor, v, ind...)
    ind = CartesianIndex(ind)
    for i in 1:length(ind)
        if ind[i]>T.size_[i]
            throw("index out of range")
        end
    end
    T.elements_[ind] = v
end
function eachindex(T::SparseTensor)
    return collect(keys(T.elements_))
end

function getindex(T::SparseTensor, inds::Array)
    res = Array{Float64}(undef, length(inds))
    for i in 1:length(inds)
        res[i] = T[inds[i]]
    end
    return res
end

function density(T::SparseTensor)
    return length(T.elements_) / prod(size(T))
end
function full(T::SparseTensor)
    if prod(size(T)) == length(T.elements_)
        return true
    elseif prod(size(T)) > length(T.elements_)
        return false
    else
        throw("should not be possible: more indices than size allows")
    end
end
function tensor(T::SparseTensor)
    @assert full(T)
    return tensor(T, "zeros")
end
function tensor(T::SparseTensor, howtofill::String)
    if howtofill=="zeros"
        tensor = zeros(size(T))
        for I in keys(T.elements_)
            tensor[I] = T.elements_[I]
        end
        return tensor
    else
        println("this fill method not implemented:  "* how)
    end
end
