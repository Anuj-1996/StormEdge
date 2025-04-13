from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import numpy as np

class StackedClassifier:
    def __init__(self):
        self.model1 = XGBClassifier()
        self.model2 = GradientBoostingClassifier()
        self.meta_model = LogisticRegression()

    def fit(self, X, y):
        tscv = TimeSeriesSplit(n_splits=5)
        meta_features = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            self.model1.fit(X_train, y_train)
            self.model2.fit(X_train, y_train)

            preds1 = self.model1.predict_proba(X_test)[:, 1]
            preds2 = self.model2.predict_proba(X_test)[:, 1]
            stacked = np.column_stack((preds1, preds2))
            meta_features.append(stacked)

        meta_X = np.vstack(meta_features)
        meta_y = y[-meta_X.shape[0]:]

        self.meta_model.fit(meta_X, meta_y)

    def predict(self, X):
        pred1 = self.model1.predict_proba(X)[:, 1]
        pred2 = self.model2.predict_proba(X)[:, 1]
        meta_input = np.column_stack((pred1, pred2))
        return self.meta_model.predict(meta_input)
