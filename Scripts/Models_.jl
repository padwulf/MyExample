using MyExample
using DelimitedFiles
using LinearAlgebra: Eigen, eigen

pre = "Data/0.5_(15, 15, 15)_(1, 1, 1)"

Y = SparseTensor(pre*"_data.txt")
K1 = readdlm(pre*"_k1.txt")
K2 = readdlm(pre*"_k2.txt")
K3 = readdlm(pre*"_k3.txt")
kernels = [K1,K2,K3]
kernelseigen = eigen.(kernels)

model = NSKRRegressor(size(Y), [10^(-2), 10^(-2), 10^(-2)], "zeros")
fit(model, kernelseigen, Y)
predict(model, kernels)

predict_LOO(model, kernelseigen, Y, (0,))
predict_LOO(model, kernelseigen, Y, (1,2,))

# LOO without indication same result as just predict (which uses the hatmatrixes,
# bit like fit_predict, but without setting the A_ coefficients (faster)).
predict(model, kernelseigen, Y) == predict_LOO(model, kernelseigen, Y, ())
