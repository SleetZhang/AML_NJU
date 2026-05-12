import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: 1-D integer arrays (0 or 1).
    Returns dict with acc, bal_acc, f1.
    """
    return {
        "acc":     accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "f1":      f1_score(y_true, y_pred, zero_division=0),
    }
