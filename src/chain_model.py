
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from .chains import compute_features, FEATURE_NAMES, NUM_FEATURES


def _fit_scaler(X: np.ndarray) -> dict:
    """Compute mean and std for each feature."""
    return {
        "mean": X.mean(axis=0).tolist(),
        "std": np.maximum(X.std(axis=0), 1e-9).tolist(),
    }


def _transform(X: np.ndarray, scaler: dict) -> np.ndarray:
    mean = np.array(scaler["mean"])
    std = np.array(scaler["std"])
    return (X - mean) / std



def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    max_iter: int = 1000,
    C: float = 1.0,
) -> dict:
    """
    Train L2-regularised logistic regression via gradient descent.

    Handles class imbalance by weighting samples inversely proportional
    to class frequency (equivalent to sklearn class_weight='balanced').

    Returns dict with 'coef' and 'intercept'.
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    # Class weights for imbalanced data
    n_pos = max(y.sum(), 1)
    n_neg = max((1 - y).sum(), 1)
    weight_pos = n / (2 * n_pos)
    weight_neg = n / (2 * n_neg)
    sample_weights = np.where(y == 1, weight_pos, weight_neg)

    for _ in range(max_iter):
        z = X @ w + b
        p = _sigmoid(z)
        error = p - y  # shape (n,)
        weighted_error = error * sample_weights

        grad_w = (X.T @ weighted_error) / n + w / C
        grad_b = weighted_error.mean()

        w -= lr * grad_w
        b -= lr * grad_b

    return {"coef": w.tolist(), "intercept": float(b)}


def _predict_proba(X: np.ndarray, model: dict) -> np.ndarray:
    """Return P(relevant) for each row."""
    w = np.array(model["coef"])
    b = float(model["intercept"])
    return _sigmoid(X @ w + b)


def _f1_score(y_true, y_pred):
    """Binary F1 score."""
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)




def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_path: str = "models/chain_model.pkl",
    scaler_path: str = "models/chain_scaler.pkl",
) -> dict[str, Any]:
    """
    Train chain relevance model on labeled feature vectors.

    Parameters
    ----------
    X : feature matrix, shape (n_pairs, 16)
    y : binary labels, 1 = relevant, 0 = irrelevant

    Returns dict with cv_f1_mean, cv_f1_std, feature_importance, n_samples.
    """
    y = y.astype(np.float64)

   
    scaler = _fit_scaler(X)
    X_scaled = _transform(X, scaler)

    
    n = len(y)
    n_folds = min(5, max(2, n // 5))
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    f1_scores = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        X_tr = X_scaled[train_idx]
        y_tr = y[train_idx]
        X_val = X_scaled[val_idx]
        y_val = y[val_idx]

        fold_model = _fit_logistic(X_tr, y_tr)
        preds = (_predict_proba(X_val, fold_model) >= 0.5).astype(int)
        f1_scores.append(_f1_score(y_val, preds))

    
    model = _fit_logistic(X_scaled, y)

    
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    
    importance = dict(zip(FEATURE_NAMES, [round(c, 4) for c in model["coef"]]))

    return {
        "cv_f1_mean": round(float(np.mean(f1_scores)), 3),
        "cv_f1_std": round(float(np.std(f1_scores)), 3),
        "feature_importance": importance,
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
    }


def load_model(
    model_path: str = "models/chain_model.pkl",
    scaler_path: str = "models/chain_scaler.pkl",
) -> tuple:
    """Load trained model and scaler. Returns (model, scaler) or (None, None)."""
    mp = Path(model_path)
    sp = Path(scaler_path)
    if mp.exists() and sp.exists():
        with open(mp, "rb") as f:
            model = pickle.load(f)
        with open(sp, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None


def predict_score(model: dict, scaler: dict, features: np.ndarray) -> float:
    """Score a single feature vector. Returns P(relevant) in [0, 1]."""
    if features.ndim == 1:
        features = features.reshape(1, -1)
    X_scaled = _transform(features, scaler)
    return float(_predict_proba(X_scaled, model)[0])


def predict_scores_batch(model: dict, scaler: dict, F: np.ndarray) -> np.ndarray:
    """Score a batch. Returns array of P(relevant)."""
    X_scaled = _transform(F, scaler)
    return _predict_proba(X_scaled, model)




def build_training_data(
    df: pd.DataFrame,
    ground_truth: list[dict],
    window_days: int = 7,
    neg_ratio: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) from ground truth.

    For each entry: relevant_event_ids → y=1, random same-window events → y=0.
    """
    from datetime import timedelta

    X_list = []
    y_list = []
    id_set = set(df["GLOBALEVENTID"].values)

    for gt in ground_truth:
        anchor_id = gt["anchor_event_id"]
        relevant_ids = set(gt.get("relevant_event_ids", []))
        if anchor_id not in id_set or not relevant_ids:
            continue

        anchor_row = df[df["GLOBALEVENTID"] == anchor_id].iloc[0]
        anchor_date = anchor_row["event_date"]
        if pd.isna(anchor_date):
            continue

        start = anchor_date - timedelta(days=window_days)
        end = anchor_date + timedelta(days=window_days)
        pool = df[
            (df["event_date"] >= start) & (df["event_date"] <= end)
            & (df["GLOBALEVENTID"] != anchor_id)
        ]

        # Positives
        positives = pool[pool["GLOBALEVENTID"].isin(relevant_ids)]
        for _, cand in positives.iterrows():
            X_list.append(compute_features(anchor_row, cand))
            y_list.append(1)

        # Negatives
        negatives = pool[~pool["GLOBALEVENTID"].isin(relevant_ids)]
        n_neg = min(len(positives) * neg_ratio, len(negatives))
        if n_neg > 0:
            for _, cand in negatives.sample(n=n_neg, random_state=42).iterrows():
                X_list.append(compute_features(anchor_row, cand))
                y_list.append(0)

    if not X_list:
        return np.zeros((0, NUM_FEATURES)), np.zeros(0)
    return np.array(X_list), np.array(y_list)


def train_from_ground_truth(
    df: pd.DataFrame,
    ground_truth_path: str,
    model_path: str = "models/chain_model.pkl",
    scaler_path: str = "models/chain_scaler.pkl",
) -> dict[str, Any]:
    """End-to-end: load ground truth → build features → train → save."""
    with open(ground_truth_path) as f:
        gt_data = json.load(f)

    chain_gt = gt_data.get("chain_ground_truth", [])
    if not chain_gt:
        return {"error": "No chain ground truth entries found"}

    X, y = build_training_data(df, chain_gt)
    if len(X) < 10:
        return {"error": f"Too few training pairs ({len(X)}). Need at least 10."}

    result = train_model(X, y, model_path, scaler_path)
    print(f"  Chain model trained: CV F1 = {result['cv_f1_mean']:.3f} ± {result['cv_f1_std']:.3f}")
    print(f"  Samples: {result['n_positive']} positive, {result['n_negative']} negative")
    return result
