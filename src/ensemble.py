"""Weighted ensemble combining RF, GBM, and NN predictions."""
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def ensemble_predict(rf_proba, gbm_proba, nn_proba, weights=None):
    """Combine predictions using weighted averaging.
    
    Final prediction: y = alpha1 * RF + alpha2 * GBM + alpha3 * NN
    """
    if weights is None:
        weights = [0.3, 0.35, 0.35]  # Default weights

    alpha1, alpha2, alpha3 = weights
    combined = alpha1 * rf_proba + alpha2 * gbm_proba + alpha3 * nn_proba
    return (combined >= 0.5).astype(int)


def optimize_weights(rf_proba, gbm_proba, nn_proba, y_true):
    """Grid search over weight combinations to find optimal ensemble weights."""
    best_acc = 0
    best_weights = None

    for a1 in np.arange(0.1, 0.6, 0.05):
        for a2 in np.arange(0.1, 0.6, 0.05):
            a3 = 1.0 - a1 - a2
            if a3 < 0.1:
                continue
            preds = ensemble_predict(rf_proba, gbm_proba, nn_proba, [a1, a2, a3])
            acc = accuracy_score(y_true, preds)
            if acc > best_acc:
                best_acc = acc
                best_weights = [a1, a2, a3]

    print(f"Optimal weights: RF={best_weights[0]:.2f}, GBM={best_weights[1]:.2f}, NN={best_weights[2]:.2f}")
    print(f"Ensemble accuracy: {best_acc:.4f}")
    return best_weights
