"""Evaluation metrics for supply chain risk prediction."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             classification_report, confusion_matrix)


def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    """Compute accuracy, F1, and MSE."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    mse = mean_squared_error(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"{model_name} Results:")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MSE:      {mse:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")

    return {'accuracy': acc, 'f1': f1, 'mse': mse}
