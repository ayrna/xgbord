import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator


class XGB(BaseEstimator):
    def __init__(
        self,
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.booster = None

    def fit(self, X, y):
        dmatrix_train = xgb.DMatrix(X, label=y)
        self.booster = xgb.train(
            {
                "tree_method": "hist",
                "disable_default_eval_metric": True,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "seed": self.random_state,
                "objective": "multi:softmax",
                "num_class": len(np.unique(y)),
            },
            dtrain=dmatrix_train,
            num_boost_round=self.n_estimators,
        )

        return self

    def predict(self, X):
        dmatrix_test = xgb.DMatrix(X)
        return self.booster.predict(dmatrix_test).astype(int)
