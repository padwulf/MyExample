using ROC
using Statistics

auc_(scores,labels) = AUC(roc(scores, labels))

function R2_(scores, labels)
    dY = scores -labels
    Ysc = labels - mean(labels)*ones(size(labels))
    return 1 - (sum(dY.*dY) / sum(Ysc.*Ysc))
end


"""
#R2_(scores, labels) = 1 - sum((scores-labels).*scores-labels)

function explained_variance_score(y_true, y_pred)
    return(sum((y_pred .- mean(y_true)) .^ 2))
end

function total_variance_score(y_true, y_pred)
    return(sum((y_true .- mean(y_true)) .^ 2))
end

function r2_score(y_true, y_pred)
    #println(y_true)
    #println(y_pred)
    println((explained_variance_score(y_true, y_pred), total_variance_score(y_true, y_pred)))
    return(explained_variance_score(y_true, y_pred) / total_variance_score(y_true, y_pred))
end
"""
