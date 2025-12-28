from pathlib import Path

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from src.features import add_total_hours, is_restaurant

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

import joblib

df = pd.read_parquet(INTERIM_DATA_DIR / 'business.parquet')
df = add_total_hours(df)
df = df.drop(columns=['longitude', 'latitude', 'business_id', 'name', 'address', 'attributes', 'hours',])

X = df.drop(columns=["is_open"])
y = df["is_open"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=320)

logit = LogisticRegression(class_weight="balanced", max_iter=1000, penalty="l2")

def ravel_1d(X):
    return np.asarray(X).ravel()

def to_str_array(x):
    return pd.Series(x).fillna("").astype(str).to_numpy()

def yelp_cat_tokenizer(s: str):
    return [t for t in s.split(", ") if t]


cat_pipe = Pipeline([
    ("select_1d", FunctionTransformer(ravel_1d, validate=False)),
    ("to_str", FunctionTransformer(to_str_array, validate=False)),
    ("vect", CountVectorizer(
        tokenizer=yelp_cat_tokenizer,
        preprocessor=None,
        token_pattern=None,
        binary=True,
        lowercase=False,
    )),
])

column_trans = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), ["total_hours_open"]),
        ("loc", OneHotEncoder(handle_unknown="ignore"), ["city", "state", "postal_code"]),
        ("categories", cat_pipe, ["categories"]),
    ],
    remainder="drop",
)


pipe = Pipeline([
    ('preprocessor', column_trans),
    ('logreg', logit)
])

param_grid = {
    "logreg__C": [0.1, 1, 10],
    'logreg__penalty': ['l2'],
    'logreg__class_weight': [None, 'balanced'],  
    "preprocessor__categories__vect__min_df": [5, 20, 100],
}

grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=StratifiedKFold(10), scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

joblib.dump(best_model, MODELS_DIR / "logistic_regression.joblib")
 