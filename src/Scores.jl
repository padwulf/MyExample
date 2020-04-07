using ROC

auc_(scores,labels) = AUC(roc(scores, labels))
R2_(scores, labels) = r2_score(labels, scores)


function explained_variance_score(y_true, y_pred)
    return(sum((y_pred .- mean(y_true)) .^ 2))
end

function total_variance_score(y_true, y_pred)
    return(sum((y_true .- mean(y_true)) .^ 2))
end

function r2_score(y_true, y_pred)
    #println(y_true)
    #println(y_pred)
    return(explained_variance_score(y_true, y_pred) / total_variance_score(y_true, y_pred))
end
