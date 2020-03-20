using MyExample

# load a fully observed dataset
data =  "Data/0.15_(15, 15, 15)_(1, 1, 1)_data.txt"
Y = SparseTensor(data)

Y
density(Y)
full(Y)
tensor(Y)
eachindex(Y)


### sparsetensor <: AbstractArray, also multiplying with MPS works, but slower
@time Mps(randn(15,15), randn(15,15), randn(15,15))*Y
Y2 = tensor(Y)
@time Mps(randn(15,15), randn(15,15), randn(15,15))*Y2


# removing some elements form Y
for i in 1:2000
    els = delete!(Y.elements_, collect(keys(Y.elements_))[1])
end
Y
density(Y)
full(Y)
tensor(Y, "zeros")   #note that here to have a tensor, we must define how to fill missing elements
