"""
Model Utilities — Federated Learning UCI HAR
============================================
Helper functions untuk serialisasi, deserialisasi, dan evaluasi model.
Digunakan bersama oleh server dan client.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, log_loss
)

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def serialize_model(model: LogisticRegression) -> dict:
    """
    Konversi model sklearn ke dictionary JSON-serializable.

    Returns
    -------
    dict dengan key: coef, intercept
    """
    return {
        "coef"      : model.coef_.tolist(),
        "intercept" : model.intercept_.tolist(),
    }


def deserialize_model(params: dict,
                      n_classes: int = 6,
                      n_features: int = 561,
                      max_iter: int = 200) -> LogisticRegression:
    """
    Buat model sklearn dari dictionary parameter.

    Parameters
    ----------
    params    : dict dengan key 'coef' dan 'intercept'
    n_classes : jumlah kelas
    n_features: jumlah fitur

    Returns
    -------
    LogisticRegression yang sudah di-set parameter-nya
    """
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=max_iter,
        warm_start=True,
        C=1.0,
        random_state=42,
    )
    model.coef_          = np.array(params["coef"])
    model.intercept_     = np.array(params["intercept"])
    model.classes_       = np.arange(1, n_classes + 1)
    model.n_features_in_ = n_features
    return model


def evaluate_model(model: LogisticRegression,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   verbose: bool = True) -> dict:
    """
    Evaluasi model pada data test.

    Returns
    -------
    dict berisi accuracy, f1_macro, f1_weighted, loss, confusion_matrix
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = {
        "accuracy"    : float(accuracy_score(y_test, y_pred)),
        "f1_macro"    : float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted" : float(f1_score(y_test, y_pred, average="weighted")),
        "log_loss"    : float(log_loss(y_test, y_prob)),
        "n_samples"   : len(y_test),
    }

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"  EVALUASI MODEL")
        print(f"{'=' * 55}")
        print(f"  Accuracy      : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
        print(f"  F1 Macro      : {metrics['f1_macro']:.4f}")
        print(f"  F1 Weighted   : {metrics['f1_weighted']:.4f}")
        print(f"  Log Loss      : {metrics['log_loss']:.4f}")
        print(f"  Test Samples  : {metrics['n_samples']}")
        print(f"\n{classification_report(y_test, y_pred, target_names=list(ACTIVITY_LABELS.values()))}")

    return metrics


def compute_label_distribution(y: np.ndarray) -> dict:
    """Hitung distribusi label dalam persentase."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {
        ACTIVITY_LABELS[int(label)]: {
            "count"  : int(count),
            "percent": float(count / total * 100)
        }
        for label, count in zip(unique, counts)
    }


def fedavg_manual(list_of_params: list, weights: list = None) -> dict:
    """
    Implementasi FedAvg secara manual (untuk keperluan edukasi/challenge).

    Parameters
    ----------
    list_of_params : list of dict {"coef": [...], "intercept": [...]}
    weights        : list of float (proporsi data per client)
                     Jika None, gunakan rata-rata sederhana (uniform)

    Returns
    -------
    dict {"coef": [...], "intercept": [...]}
    """
    n_clients = len(list_of_params)

    if weights is None:
        weights = [1.0 / n_clients] * n_clients
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    avg_coef      = None
    avg_intercept = None

    for params, w in zip(list_of_params, weights):
        coef      = np.array(params["coef"])
        intercept = np.array(params["intercept"])

        if avg_coef is None:
            avg_coef      = w * coef
            avg_intercept = w * intercept
        else:
            avg_coef      += w * coef
            avg_intercept += w * intercept

    return {
        "coef"      : avg_coef.tolist(),
        "intercept" : avg_intercept.tolist(),
    }
