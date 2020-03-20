using ROC

auc_(scores,labels) = AUC(roc(scores, labels))
