"""Random Forest with optimized tree pruning and feature importance weighting."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def build_random_forest(X_train, y_train):
    """Build RF with optimized pruning via grid search on max_depth/min_samples."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best RF params: {grid_search.best_params_}")
    print(f"Best RF CV accuracy: {grid_search.best_score_:.4f}")

    return best_model


def get_feature_importance(model, feature_names=None):
    """Extract and rank feature importances using Gini impurity."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    if feature_names is not None:
        return [(feature_names[i], importances[i]) for i in indices[:20]]
    return [(i, importances[i]) for i in indices[:20]]
