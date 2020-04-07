module MyExample




export NSKRRegressor, KKRegressor
export fit,predict,predict_LOO,fit_iteratively


export SparseTensor
export readtensor

export Kfold, CVestimate, CVscore, optimizeHyperParameters

export auc_,R2_

export SE, self, L2


#include("NSKRRegressor.jl")
include("ModelSelection.jl")
#include("")
#include("MatrixProductState.jl")


#moeten uiteindelijk niet echt geexporteerd worden:
export Mps
export optimalmul





end # module
