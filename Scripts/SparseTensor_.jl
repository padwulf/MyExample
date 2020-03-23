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
Y2 = SparseTensor(Y, eachindex(Y)[1:1000])
density(Y2)
full(Y2)
tensor(Y, "zeros")
