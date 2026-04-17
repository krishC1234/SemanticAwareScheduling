#!/usr/bin/env python3
"""Train a few candidate regressors to predict the scaling exponent k from
data/scaling_dataset.csv and report cross-validated losses for each.

Models compared:
  - Ridge        : linear baseline (with log(param_count))
  - RandomForest : non-linear, robust to feature scales
  - GradBoosting : typically strongest on small tabular data

Loss metrics reported per 5-fold CV: MAE, RMSE.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

HERE = Path(__file__).parent
CSV = HERE.parent / "train_data" / "scaling_dataset.csv"
RESULTS_TXT = HERE / "results.txt"
FEATURES_JSON = HERE / "feature_columns.json"
N_SPLITS = 5
SEED = 0


def load():
    df = pd.read_csv(CSV)
    y = df.pop("k").to_numpy()
    groups = df.pop("model_name").to_numpy()  # used only for GroupKFold
    X = df  # keep DataFrame so column names survive into the pipeline
    return X, y, groups


def log_param_count_transformer(numeric_cols):
    """Log-transform + standardize the numeric columns; passthrough one-hots."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("log", FunctionTransformer(np.log1p, validate=False)),
                ("scale", StandardScaler()),
            ]), numeric_cols),
        ],
        remainder="passthrough",
    )


def build_models(numeric_cols):
    return {
        "Ridge": Pipeline([
            ("prep", log_param_count_transformer(numeric_cols)),
            ("reg",  Ridge(alpha=1.0, random_state=SEED)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=SEED, n_jobs=-1,
        ),
        "GradBoosting": GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED,
        ),
        "MLP_32": Pipeline([
            ("prep", log_param_count_transformer(numeric_cols)),
            ("reg",  MLPRegressor(
                hidden_layer_sizes=(32,), activation="relu",
                solver="adam", learning_rate_init=1e-3, alpha=1e-3,
                max_iter=150, random_state=SEED,
            )),
        ]),
    }


def cv_losses(model, X, y, groups, kf):
    maes, rmses = [], []
    for tr, te in kf.split(X, y, groups=groups):
        Xtr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        Xte = X.iloc[te] if hasattr(X, "iloc") else X[te]
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        err = pred - y[te]
        maes.append(np.mean(np.abs(err)))
        rmses.append(np.sqrt(np.mean(err ** 2)))
    return np.array(maes), np.array(rmses)


def main():
    X, y, groups = load()
    n_models = len(np.unique(groups))
    feature_cols = list(X.columns)
    numeric_cols = ["batch_size", "param_count"]

    # Buffer everything we print so we can also dump it to results.txt.
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"loaded {len(y)} rows across {n_models} unique models, " f"{X.shape[1]} features ({feature_cols})")
    emit()

    # GroupKFold: every model_name lives in exactly one fold -> no leakage of "same model, different config" rows from train into test.
    kf = GroupKFold(n_splits=N_SPLITS)

    emit(f"{N_SPLITS}-fold GroupKFold CV losses (grouped by model_name):")
    emit(f"  {'model':<14} {'MAE':>16} {'RMSE':>16}")
    emit(f"  {'-'*14} {'-'*16} {'-'*16}")

    cv_scores = {}
    for name, model in build_models(numeric_cols).items():
        mae, rmse = cv_losses(model, X, y, groups, kf)
        cv_scores[name] = (mae.mean(), rmse.mean())
        emit(f"  {name:<14} "
             f"{mae.mean():.4f} +/- {mae.std():.4f}   "
             f"{rmse.mean():.4f} +/- {rmse.std():.4f}")

    # Honest baseline: per fold, predict the training fold's mean on the test
    # fold. Different from in-sample MAE(y, y.mean()) because train/test means
    # diverge under GroupKFold (some folds hold out higher-k families).
    base_mae, base_rmse = [], []
    for tr, te in kf.split(X, y, groups=groups):
        pred = np.full_like(y[te], y[tr].mean(), dtype=float)
        err = pred - y[te]
        base_mae.append(np.mean(np.abs(err)))
        base_rmse.append(np.sqrt(np.mean(err ** 2)))
    emit()
    emit(f"  {'predict-mean':<14} "
         f"{np.mean(base_mae):.4f} +/- {np.std(base_mae):.4f}   "
         f"{np.mean(base_rmse):.4f} +/- {np.std(base_rmse):.4f}"
         f"   <- honest grouped-CV baseline")

    # Permutation test: shuffle y, re-run RF CV. If real RF beats this by a
    # lot, the model is finding signal. If it's close, the "improvement" over
    # baseline is noise.
    rng = np.random.default_rng(SEED)
    y_shuf = rng.permutation(y)
    rf = build_models(numeric_cols)["RandomForest"]
    perm_mae, perm_rmse = cv_losses(rf, X, y_shuf, groups, kf)
    emit(f"  {'RF on shuffled':<14} "
         f"{perm_mae.mean():.4f} +/- {perm_mae.std():.4f}   "
         f"{perm_rmse.mean():.4f} +/- {perm_rmse.std():.4f}"
         f"   <- noise floor; RF lift over this = real signal")

    # Pick the model with the lowest CV MAE, refit on ALL the data (CV was
    # only for evaluation), and save it. The saved Pipeline includes its own
    # preprocessing, so callers just pass a DataFrame with the columns listed
    # in feature_columns.json.
    best_name = min(cv_scores, key=lambda n: cv_scores[n][0])
    best_model = build_models(numeric_cols)[best_name] 
    best_model.fit(X, y)
    best_path = HERE / "best_model.joblib"
    joblib.dump(best_model, best_path)
    emit()
    emit(f"Best model: {best_name} (CV MAE {cv_scores[best_name][0]:.4f})")
    emit(f"  saved {best_path.name}")

    FEATURES_JSON.write_text(json.dumps({
        "feature_columns": feature_cols,
        "numeric_cols": numeric_cols,
        "target": "k",
    }, indent=2))
    emit(f"  saved {FEATURES_JSON.name}")

    RESULTS_TXT.write_text("\n".join(lines) + "\n")
    print(f"\nresults written to {RESULTS_TXT}")

if __name__ == "__main__":
    main()