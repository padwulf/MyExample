import MLBase

# regression metrics
function R2_(scores, labels)
    dY = scores -labels
    Ysc = labels - mean(labels)*ones(size(labels))
    return 1 - (sum(dY.*dY) / sum(Ysc.*Ysc))
end

# Classification metrics
function auc_precisionrecall(scores, labels)
    r=MLBase.roc(labels, scores)
    pr, rec = precision_recall(r)
    return area_under_curve(rec, pr)
end
function auc_roc(scores, labels)
    r=MLBase.roc(labels, scores)
    tpr, fpr = TPR_FPR(r)
    return area_under_curve(fpr, tpr)
end
function max_f1(scores, labels)
    r=MLBase.roc(labels, scores)
    return maximum(harmonicmeans(r))
end

#####
# Support functions
function precision_recall(r)
    pr = []
    recall = []
    for i in 1:length(r)
        push!(pr, MLBase.precision(r[i]))
        push!(recall, MLBase.recall(r[i]))
    end
    return(pr, recall)
end
function TPR_FPR(r)
    TPR = []
    FPR = []
    for i in 1:length(r)
        push!(TPR, MLBase.true_positive_rate(r[i]))
        push!(FPR, MLBase.false_positive_rate(r[i]))
    end
    return(TPR, FPR)
end
function harmonicmeans(r)
    hmeans = []
    for i in 1:length(r)
        push!(hmeans, MLBase.f1score(r[i]))
    end
    return hmeans
end
function area_under_curve(x,y)
    @assert length(x)==length(y)
    A = 0
    for i in 1:length(x)-1
        y_= (y[i]+y[i+1])/2
        dx= x[i+1]-x[i]
        A+=y_*dx
    end
    return abs(A)
end
