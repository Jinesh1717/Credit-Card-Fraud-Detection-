from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from .config import FALSE_ALARM_COST, MISSED_FRAUD_COST

def threshold_cost(y, p, threshold, amounts=None):
    pred = p >= threshold; fp = ((pred==1)&(y==0)).sum(); missed = (y[(pred==0)&(y==1)].sum())
    fraud_cost = amounts[(pred==0)&(y==1)].sum() if amounts is not None else missed*MISSED_FRAUD_COST
    return fp*FALSE_ALARM_COST + fraud_cost

def choose_threshold(y, p, amounts=None):
    ts = np.linspace(.01, .99, 99); costs = [threshold_cost(y, p, t, amounts) for t in ts]
    return float(ts[int(np.argmin(costs))])

def metrics(name, y, p, threshold=.5):
    pred = p >= threshold; tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {"model":name,"threshold":threshold,"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"f2":fbeta_score(y,pred,beta=2,zero_division=0),"pr_auc":average_precision_score(y,p),"roc_auc":roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan,"mcc":matthews_corrcoef(y,pred),"false_positives":fp,"false_negatives":fn,"tn":tn,"tp":tp}
