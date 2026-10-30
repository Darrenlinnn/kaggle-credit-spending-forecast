"""
Credit card monthly spending forecast — final stacking pipeline.

Reproduces the best Kaggle submission (public leaderboard RMSE 247.41,
rank 50 of 354) from Columbia APAN 5200 Applied Machine Learning.

Usage:
    python src/train.py --train data/analysis_data.csv --score data/scoring_data.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "monthly_spend"
ID_COL = "customer_id"
EPS = 1e-6


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7 engineered features: spend proxy, ratios, interactions, polynomial."""
    out = df.copy()
    # Direct spend proxy: number of transactions x average transaction value
    out["estimated_spend"] = out["num_transactions"] * out["avg_transaction_value"]
    # Ratios and interactions
    out["limit_to_income_ratio"] = out["credit_limit"] / (out["annual_income"] + EPS)
    out["score_x_limit"] = out["credit_score"] * out["credit_limit"]
    out["age_x_income"] = out["age"] * out["annual_income"]
    out["online_tx_ratio"] = out["online_shopping_freq"] / (out["num_transactions"] + EPS)
    out["tx_per_card"] = out["num_transactions"] / (out["num_credit_cards"] + EPS)
    # Polynomial term
    out["credit_score_sq"] = out["credit_score"] ** 2
    return out


def build_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    """Preprocessing + stacking ensemble (RF, GB, Ridge -> LassoCV)."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, categorical_cols),
    ])

    base_models = [
        ("rf", RandomForestRegressor(n_estimators=150, max_depth=7,
                                     random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=220, learning_rate=0.07,
                                         max_depth=6, subsample=0.8,
                                         random_state=42)),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ]
    # LassoCV meta-model blends base predictions with regularization
    stacker = StackingRegressor(
        estimators=base_models,
        final_estimator=LassoCV(alphas=[0.01, 0.1, 1, 10], cv=5, random_state=42),
        cv=5,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("stacker", stacker)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/analysis_data.csv")
    parser.add_argument("--score", default="data/scoring_data.csv")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    train = create_features(pd.read_csv(args.train))
    score = create_features(pd.read_csv(args.score))

    features = [c for c in train.columns if c not in (ID_COL, TARGET)]
    numeric_cols = train[features].select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in features if c not in numeric_cols]

    X, y = train[features], train[TARGET]
    model = build_pipeline(numeric_cols, categorical_cols)

    # Baseline: always predict the training mean
    baseline_rmse = np.sqrt(np.mean((y - y.mean()) ** 2))
    print(f"Baseline RMSE (predict mean): {baseline_rmse:.2f}")

    if not args.skip_cv:
        # 5-fold cross-validation for a stable error estimate
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse = -cross_val_score(model, X, y, cv=kf,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
        print(f"5-fold CV RMSE: {rmse.mean():.2f} ± {rmse.std():.2f}")

    # Fit on all training data and write predictions
    model.fit(X, y)
    pd.DataFrame({ID_COL: score[ID_COL],
                  TARGET: model.predict(score[features])}).to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
