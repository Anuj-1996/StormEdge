import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score
import logging

class TrendPredictor:
    def __init__(self, tune: bool = False):
        self.tune = tune
        self.model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.fitted = False

    def train(self, df: pd.DataFrame, feature_cols: list, target_col: str) -> None:
        X = df[feature_cols]
        y = df[target_col]

        tscv = TimeSeriesSplit(n_splits=5)

        if self.tune:
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            }

            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                scoring='accuracy',
                cv=tscv,
                verbose=1,
                n_jobs=-1
            )

            logging.info("[INFO] Starting hyperparameter tuning...")
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            logging.info(f"[INFO] Best parameters: {grid_search.best_params_}")
            logging.info(f"[INFO] Best CV Score: {grid_search.best_score_:.4f}")
        else:
            for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                logging.info(f"[INFO] Fold {fold} Accuracy: {acc:.4f}")

        self.fitted = True

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        if not self.fitted:
            raise Exception("Model not trained yet.")
        return self.model.predict(df[feature_cols])

    def predict_proba(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        if not self.fitted:
            raise Exception("Model not trained yet.")
        return self.model.predict_proba(df[feature_cols])
