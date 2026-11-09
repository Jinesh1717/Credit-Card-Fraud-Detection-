from __future__ import annotations
import logging, joblib, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:  # keeps schema/scoring tests runnable before optional install
    ImbPipeline = Pipeline
    class _NoOp:
        def __init__(self, *args, **kwargs): pass
        def fit(self, X, y=None): return self
        def transform(self, X): return X
        def fit_resample(self, X, y): return X, y
    SMOTE = RandomUnderSampler = _NoOp
from .config import MODEL_DIR, RANDOM_STATE
from .data import load_data, split_data
from .features import make_preprocessor
from .evaluate import metrics, choose_threshold

def build_models():
    pre = make_preprocessor()
    return {
      "dummy": Pipeline([("pre",pre),("model",DummyClassifier(strategy="most_frequent"))]),
      "logistic_balanced": Pipeline([("pre",pre),("model",LogisticRegression(max_iter=500,class_weight="balanced",random_state=RANDOM_STATE))]),
      "logistic_smote": ImbPipeline([("pre",pre),("smote",SMOTE(random_state=RANDOM_STATE)),("model",LogisticRegression(max_iter=500,random_state=RANDOM_STATE))]),
      "random_forest": Pipeline([("pre",pre),("model",RandomForestClassifier(n_estimators=120,class_weight="balanced_subsample",n_jobs=-1,random_state=RANDOM_STATE))]),
      "random_forest_under": ImbPipeline([("pre",pre),("under",RandomUnderSampler(random_state=RANDOM_STATE)),("model",RandomForestClassifier(n_estimators=120,n_jobs=-1,random_state=RANDOM_STATE))]),
      "isolation_forest": Pipeline([("pre",pre),("model",IsolationForest(n_estimators=100,contamination="auto",random_state=RANDOM_STATE))]),
    }

def main():
    logging.basicConfig(level=logging.INFO); df, synthetic = load_data(); Xtr,Xte,ytr,yte = split_data(df)
    rows=[]; MODEL_DIR.mkdir(exist_ok=True)
    for name, model in build_models().items():
        model.fit(Xtr,ytr); p = model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else -model.decision_function(Xte)
        p = (p-p.min())/(p.max()-p.min()+1e-12); t=choose_threshold(ytr, (model.predict_proba(Xtr)[:,1] if hasattr(model,"predict_proba") else p[:len(ytr)]) ) if name != "isolation_forest" else .5
        rows.append(metrics(name,yte,p,t)); joblib.dump({"model":model,"threshold":t,"synthetic":synthetic}, MODEL_DIR/f"{name}.joblib")
    out=pd.DataFrame(rows).sort_values("pr_auc",ascending=False); out.to_csv("reports/metrics.csv",index=False); out.to_json("reports/model_summary.json",orient="records",indent=2); print(out[["model","pr_auc","precision","recall","threshold"]].to_string(index=False))
if __name__ == "__main__": main()
