from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

def make_preprocessor():
    scaled = ["Time", "Amount"]
    return ColumnTransformer([( "scaled", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())]), scaled)], remainder="passthrough")
