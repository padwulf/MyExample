include("MatrixProductState.jl")
include("SparseTensor.jl")

#identity activation
self(x) = x
self_deriv(x) = 1
#sigmoid activation
sigmoid(x) = 1 / (1+exp(-x))
sigmoid_deriv(x) = sigmoid(x)*(1-sigmoid(x))
#binaray crossentropy loss
bce(y, f) = -y*log(f+eps()) - (1-y)*log(1-f+eps())
bce_deriv(y,f) = -y/(f+eps()) + (1-y)/(1-f+eps())
#squared error loss
se(y,f) = (f-y)^2
se_deriv(y,f) = 2*(f-y)
# l2 regularization
l2(A,K) = sum(A .* (A*K))
l2_deriv(A,K) = 2*(A*K)

# Loss: generic for all combinations in losses and regs: F and G are value and derivative w.r.t. A
# l = given as argument = l(y,f)  with f = a(A*K)
# r = given as argumnet = r(A,K)
# note that Y is a dense label tensor. inds denote what labels are really observed to evaluate the lossfunction.
function Loss_reg!(F,G,A,K,act,actderiv,λ,Y,inds, loss,lossderiv, reg, regderiv)
    z = A*K
    f = act.(z)
    if G!=nothing
        dldf_dfdz = zeros(size(f))
        for I in inds
            dldf_dfdz[I] = lossderiv(Y[I], f[I]) * actderiv(z[I])
        end
        dldf_dfdz_dzdA = dldf_dfdz * K
        dr_dA = regderiv(A,K)
        G[:] = dldf_dfdz_dzdA + λ * dr_dA
    end
    if F != nothing
        l = 0.
        for I in inds
            l+= loss(Y[I],f[I])
        end
        return l + λ * reg(A,K)
    end
end

# Loss: lossfunction for n-step kernel ridge regression: F and G are value and derivative w.r.t. A
# l = SE
# r = f * (H^(-1) - 1) * f
function NSKRR_loss!(F,G,A,K,inverseH, Y, inds)
    f = A*K
    if G!=nothing
        dldf = zeros(size(Y))
        for I in inds
            dldf[I] = 2*(f[I] - Y[I])
        end
        dldf_dfdA = dldf*K
        drdA = 2*(inverseH*f -f)*K
        G[:] = dldf_dfdA + drdA
    end
    if F!=nothing
        SE = 0.
        for I in inds
            SE+= (f[I]-Y[I])^2
        end
        return SE + sum(f.*(inverseH*f -f))
    end
end


## Below losses specific implemented for rigde regression and logistic ridge regression
## However not really needed anymore due to more generic formulation above, which as nearly as efficient
"""
# Loss: squared error + L2 regularization: F and G are value and derivative w.r.t. A
# l = SE = (f -Y)^2  with f = A*K
# r = L2 = A*K*A
function SE_reg!(F,G,A,K,λ,Y,inds)
    f = A*K
    if G != nothing
        dldf = zeros(size(f))
        for I in inds
            dldf[I] = 2*(f[I]-Y[I])
        end
        dldf_dfdA = dldf*K
        dr_dA = 2*A*K
        G[:] = dldf_dfdA + λ * dr_dA
    end
    if F != nothing
        SE = 0.
        for I in inds
            SE+= (f[I]-Y[I])^2
        end
        return SE + λ * sum(A.*(A*K))
    end
end


bce(1,1)
bce(1,0)
bce_deriv(1,0)


# Loss: binary crossentropy + L2 regularization: F and G are value and derivative w.r.t. A
# l = BCE = -[ y*log(f) + (1-y)* log(1-f) ] with f = a(A*K)
# r = L2 = A*K*A
function BCE_reg!(F,G,A,K,a,aderiv,λ,Y,inds)
    z = A*K
    f = a.(z)
    if G!=nothing
        dldf_dfdz = zeros(size(f))
        for I in inds
            dldf_dfdz[I] = bce_deriv(Y[I], f[I]) * aderiv(z[I])
        end
        dldf_dfdz_dzdA = dldf_dfdz * K
        dr_dA = 2*A*K
        G[:] = dldf_dfdz_dzdA + λ * dr_dA
    end
    if F != nothing
        BCE = 0.
        for I in inds
            BCE+= bce(Y[I],f[I])
        end
        return BCE + λ * sum(A.*(A*K))
    end
end
"""
