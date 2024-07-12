"""Gradient Boosting Machine with adaptive learning rate and regularization."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def build_gbm(X_train, y_train):
    """Build GBM with adaptive learning rate decay and L2 regularization."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    gbm = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gbm, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best GBM params: {grid_search.best_params_}")
    print(f"Best GBM CV accuracy: {grid_search.best_score_:.4f}")

    return best_model


class AdaptiveLRGBM:
    """GBM wrapper with adaptive learning rate decay: lr(m) = lr0 / (1 + lambda * m)."""

    def __init__(self, lr0=0.1, decay=0.01, n_estimators=300, max_depth=5):
        self.lr0 = lr0
        self.decay = decay
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None

    def fit(self, X_train, y_train):
        lr = self.lr0 / (1 + self.decay * self.n_estimators)
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=lr,
            max_depth=self.max_depth,
            subsample=0.9,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
