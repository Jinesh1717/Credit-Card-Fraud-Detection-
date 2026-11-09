import pandas as pd
from src.data import make_synthetic, validate_schema
from src.score import score_frame
from src.train import build_models
def test_schema(): validate_schema(make_synthetic(100))
def test_scaler_fit_only_on_train():
    x=make_synthetic(100); a=build_models()["logistic_balanced"]; a.fit(x.drop(columns="Class").iloc[:80],x.Class.iloc[:80]); assert not hasattr(a.named_steps["pre"].named_transformers_["scaled"].named_steps["scale"],"data_min_")
def test_score_columns():
    x=make_synthetic(100); a=build_models()["logistic_balanced"]; a.fit(x.drop(columns="Class"),x.Class); out=score_frame(x.drop(columns="Class"),{"model":a,"threshold":.5}); assert {"fraud_probability","risk_score","risk_tier"}.issubset(out.columns)
