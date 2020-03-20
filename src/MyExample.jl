module MyExample




export NSKRRegressor
export fit,predict,predict_LOO


export SparseTensor
export density, tensor, full, fill

export Kfold, CVestimate, CVscore, optimizeHyperParameters

export auc_


#include("NSKRRegressor.jl")
include("ModelSelection.jl")
#include("")
#include("MatrixProductState.jl")


#moeten uiteindelijk niet echt geexporteerd worden:
export Mps
export optimalmul





end # module
