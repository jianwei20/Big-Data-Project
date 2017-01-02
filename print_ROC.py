import numpy as np
def roc(actual, pred):
    fpr=np.array([1.0])
    tpr=np.array([1.0])
    n=float(len(actual)-sum(actual))
    p=float(sum(actual))
    for i in np.arange(min(pred), max(pred), 1.0/len(pred)):
        TP=0.0
        FP=0.0
        for j in range(len(pred)):
            if (pred[j] > i) & (actual[j]==1):
                TP+=1
            if (pred[j] > i) & (actual[j]==0):
                FP+=1

        tpr = np.insert(tpr, 0, TP/p)
        fpr = np.insert(fpr, 0, FP/n)
    return fpr, tpr
