using Plots
using DelimitedFiles
using LinearAlgebra
using Optim
using MyExample

# Data reading

pre = "Data/0.5_(15, 15, 15)_(1, 1, 1)"
Y = SparseTensor(pre*"_data.txt")
K1 = readdlm(pre*"_k1.txt")
K2 = readdlm(pre*"_k2.txt")
K3 = readdlm(pre*"_k3.txt")
K = [K1,K2,K3]
Ke = eigen.(K)

model = NSKRRegressor(size(Y), [10^(-2), 10^(-2), 10^(-2)], "zeros")

# Testing CVestimate
                #take many folds -> near LOO. and compare k-fold with LOO. Note that in k-fold, the missing
                #test data is set to zero.

## setting
s = CVestimate(model, Ke, Y, 2000, (0,))
s2 = CVestimate(model, Ke, Y, "LOO", (0,))
plot(tensor(s2)[:], tensor(s)[:], seriestype = :scatter)
# compare in plot: relation between both, rigid on one side. Prob depends on λ

## other setting
s = CVestimate(model, Ke, Y, 3200, (1,))
s2 = CVestimate(model, Ke, Y, "LOO", (1,))
plot(tensor(s2)[:], tensor(s)[:], seriestype = :scatter)
# compare in plot: not really relation visible, complete rows set to zero

## larger lambda
model = NSKRRegressor(size(Y), [10^(-0.5), 10^(-0.5), 10^(-0.5)], "zeros")
s = CVestimate(model, Ke, Y, 3200, (1,))
s2 = CVestimate(model, Ke, Y, "LOO", (1,))
plot(tensor(s2)[:], tensor(s)[:], seriestype = :scatter)
# compare in plot: relation somewhat more visible (k-fold depends les on settozeros)



#testing modeloptimization
println("--")
## For NSKKRegressor
### Setting (0,)
model =  NSKRRegressor(size(Y),[0.1,0.1,0.1], "zeros")
modelopt1, opt1 = optimizeHyperParameters(model, Ke,Y,"LOO",(0,),auc_, LBFGS(), [0.0001,0.0001,0.0001], false)
modelopt2,opt2 = optimizeHyperParameters(model, Ke,Y,"LOO",(0,),auc_, LBFGS(), "notneeded", true)
modelopt3,opt3 = optimizeHyperParameters(model, Ke,Y,10,(0,),auc_, LBFGS(), "notneeded", true)

CVscore(model, Ke, Y, "LOO", (0,), auc_)
CVscore(modelopt1, Ke, Y, "LOO", (0,), auc_)
CVscore(modelopt2, Ke, Y, "LOO", (0,), auc_)
CVscore(modelopt2, Ke, Y, 10, (0,), auc_,5)
CVscore(modelopt3, Ke, Y, 10, (0,), auc_,5)


### Setting (1,)
model =  NSKRRegressor(size(Y),[0.1,0.1,0.1], "zeros")
modelopt1, opt1 = optimizeHyperParameters(model, Ke,Y,"LOO",(1,),auc_, LBFGS(), [0.0001,0.0001,0.0001], false)
modelopt2,opt2 = optimizeHyperParameters(model, Ke,Y,"LOO",(1,),auc_, LBFGS(), "notneeded", true)
modelopt3,opt3 = optimizeHyperParameters(model, Ke,Y,10,(1,),auc_, LBFGS(), "notneeded", true)

CVscore(model, Ke, Y, "LOO", (1,), auc_)
CVscore(modelopt1, Ke, Y, "LOO", (1,), auc_)  #did not really converge
CVscore(modelopt2, Ke, Y, "LOO", (1,), auc_) # univariate and easier converge
CVscore(modelopt2, Ke, Y, 10, (1,), auc_,5)
CVscore(modelopt3, Ke, Y, 10, (1,), auc_,5)

### Setting (1,2,)
model =  NSKRRegressor(size(Y),[0.1,0.1,0.1], "zeros")
modelopt1, opt1 = optimizeHyperParameters(model, Ke,Y,"LOO",(1,2),auc_, LBFGS(), [0.0001,0.0001,0.0001], false)
modelopt2,opt2 = optimizeHyperParameters(model, Ke,Y,"LOO",(1,2),auc_, LBFGS(), "notneeded", true)
modelopt3,opt3 = optimizeHyperParameters(model, Ke,Y,10,(1,2),auc_, LBFGS(), "notneeded", true)

CVscore(model, Ke, Y, "LOO", (1,2), auc_)
CVscore(modelopt1, Ke, Y, "LOO", (1,2), auc_)
CVscore(modelopt2, Ke, Y, "LOO", (1,2), auc_)
CVscore(modelopt2, Ke, Y, 10, (1,2), auc_,5)
CVscore(modelopt3, Ke, Y, 10, (1,2), auc_,5)

###setting (1,2,3,)
model =  NSKRRegressor(size(Y),[0.1,0.1,0.1], "zeros")
modelopt1, opt1 = optimizeHyperParameters(model, Ke,Y,"LOO",(1,2,3),auc_, LBFGS(), [0.0001,0.0001,0.0001], false)
modelopt2,opt2 = optimizeHyperParameters(model, Ke,Y,"LOO",(1,2,3),auc_, LBFGS(), "notneeded", true)
#modelopt3,opt3 = optimizeHyperParameters(model, Ke,Y,10,(1,2),auc_, LBFGS(), "notneeded", true)

CVscore(model, Ke, Y, "LOO", (1,2,3), auc_)
CVscore(modelopt1, Ke, Y, "LOO", (1,2,3), auc_)
CVscore(modelopt2, Ke, Y, "LOO", (1,2,3), auc_)
#CVscore(modelopt2, Ke, Y, 10, (1,2,3), auc_,5)
#CVscore(modelopt3, Ke, Y, 10, (1,2,3), auc_,5)



## For KKRegressor
model = KKRegressor(size(Y), [0.01], self, SE, L2)
auc_(predict(model, K), Y)
fit(model, K, Y)
auc_(predict(model, K), Y)
fit(model, K, Y)
auc_(predict(model, K), Y)

CVscore(model, K, Y, 3, (0,), auc_, 3)
CVscore(model, K, Y, 5, (0,), auc_, 3)

model2 = KKRegressor(size(Y), [1.0], self, SE, L2)
CVscore(model2, K, Y, 5, (0,), auc_, 3)
CVscore(model2, K, Y, 10, (1,), auc_, 3)
CVscore(model2, K, Y, 5, (1,2,), auc_, 3)
CVscore(model2, K, Y, 5, (2,3,), auc_, 3)
