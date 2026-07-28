import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import StratifiedShuffleSplit
from models._utils import stable_inverse_logistic_function, get_color_palete_from_n_classes
from sklearn.base import BaseEstimator
from sklearn.utils.class_weight import compute_class_weight


class XGBORD(BaseEstimator):
    def __init__(
        self,
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        lbfgs_optimizer=False,
        optimize_thresholds_learning_rate="same",
        balanced_class_weights=False,
        thresholds_method="optimal",
        random_state=None,
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.lbfgs_optimizer = lbfgs_optimizer
        self.thresholds_method = thresholds_method
        self.optimize_thresholds_learning_rate = optimize_thresholds_learning_rate
        self.balanced_class_weights = balanced_class_weights
        self.random_state = random_state

        self.booster = None

    def _setup(self, y):
        y = y.astype(int)
        self._classes, counts = np.unique(y, return_counts=True)

        if self.balanced_class_weights:
            print("Using balanced class weights.")
            self.class_weights = compute_class_weight(
                class_weight="balanced", classes=self._classes, y=y
            )
        else:
            print("Using uniform class weights.")
            self.class_weights = np.ones(len(self._classes))

        assert (
            np.arange(len(self._classes)).all() == self._classes.all()
        ), "y must contain all classes from 0 to Q-1"

        assert self.thresholds_method in [
            "none",
            "optimal",
            "imp_med",
            "mean",
            "lat_med",
        ], "thresholds_methods must be 'none', 'optimal', 'median' or 'mean'"

        self._n_classes = len(self._classes)

        rel_freq = counts / counts.sum()

        self.thresholds = np.zeros(len(self._classes) - 1)
        for q in range(0, len(self._classes) - 1):
            cumprob = rel_freq[: q + 1].sum()
            self.thresholds[q] = stable_inverse_logistic_function(cumprob)

        assert (
            self.thresholds[1:] > self.thresholds[:-1]
        ).all(), "thresholds must be in ascending order"

        return self

    def optimize_thresholds(self, X, y):
        if self.lbfgs_optimizer:
            print("Optimizing thresholds using L-BFGS...")
            from scipy.optimize import minimize

            if self.booster is None:
                raise ValueError("Model must be fitted before optimizing thresholds.")

            dmatrix_test = xgb.DMatrix(X)
            projections = self.booster.predict(dmatrix_test)
            y_int = y.astype(int)
            n_thresholds = self._n_classes - 1

            def objective_and_grad(thresholds):
                loss = 0.0
                grad = np.zeros(n_thresholds)

                for i, y_i in enumerate(y_int):
                    f_i = projections[i]
                    for q in range(n_thresholds):
                        if q < y_i:
                            # Should be f(x) > theta_q, i.e. sigma(f - theta) -> 1
                            z = f_i - thresholds[q]
                            sig = 1.0 / (1.0 + np.exp(-z))
                            loss -= np.log(sig + 1e-15)
                            grad[q] += sig - 1  # = -1/(1+exp(z)) = -(1 - sig)
                        else:
                            # Should be f(x) <= theta_q, i.e. sigma(theta - f) -> 1
                            z = thresholds[q] - f_i
                            sig = 1.0 / (1.0 + np.exp(-z))
                            loss -= np.log(sig + 1e-15)
                            grad[q] -= sig - 1  # = +1/(1+exp(z)) = (1 - sig)

                return loss, grad

            result = minimize(
                objective_and_grad,
                x0=self.thresholds.copy(),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-8},
            )

            self.thresholds = result.x
        else:
            if self.booster is None:
                raise ValueError("Model must be fitted before optimizing thresholds.")
            if self.optimize_thresholds_learning_rate == "same":
                ot_learning_rate = self.learning_rate
            else:
                ot_learning_rate = self.optimize_thresholds_learning_rate

            dmatrix_test = xgb.DMatrix(X)
            projections = self.booster.predict(dmatrix_test)
            for i, y_i in enumerate(y.astype(int)):
                for q in range(y_i):
                    thresholds_q_grad = 1.0 / (
                        1.0 + np.exp(projections[i] - self.thresholds[q])
                    )
                    self.thresholds[q] = self.thresholds[q] - (
                        ot_learning_rate * thresholds_q_grad
                    )
                for q in range(y_i, self._n_classes - 1):
                    thresholds_q_grad = -1.0 / (
                        1.0 + np.exp(self.thresholds[q] - projections[i])
                    )
                    self.thresholds[q] = self.thresholds[q] - (
                        ot_learning_rate * thresholds_q_grad
                    )

        return self

    def fit(self, X, y):
        self._setup(y)

        custom_obj = make_custom_obj(
            Q=self._n_classes, thresholds=self.thresholds, class_weights=self.class_weights
        )

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
            },
            dtrain=dmatrix_train,
            num_boost_round=self.n_estimators,
            obj=custom_obj,
        )
        if self.thresholds_method == "optimal":
            print("Optimizing thresholds...")
            self.optimize_thresholds(X, y)
        elif self.thresholds_method == "imp_med":
            print("Computing thresholds using median projections...")
            if isinstance(X, (np.ndarray, list)):
                X_df = pd.DataFrame(X)
            X_df["y"] = y.astype(int)
            median_patterns_per_class = X_df.groupby("y").median().reset_index(drop=True)
            median_projections_per_class = self.booster.predict(
                xgb.DMatrix(median_patterns_per_class)
            )
            self.thresholds = (
                median_projections_per_class[:-1] + median_projections_per_class[1:]
            ) / 2.0
        elif self.thresholds_method == "mean":
            print("Computing thresholds using mean projections...")
            if isinstance(X, (np.ndarray, list)):
                X_df = pd.DataFrame(X)
            X_df["y"] = y.astype(int)
            mean_patterns_per_class = X_df.groupby("y").mean().reset_index(drop=True)
            mean_projections_per_class = self.booster.predict(
                xgb.DMatrix(mean_patterns_per_class)
            )
            self.thresholds = (
                mean_projections_per_class[:-1] + mean_projections_per_class[1:]
            ) / 2.0
        elif self.thresholds_method == "lat_med":
            projections = self.booster.predict(dmatrix_train)
            mean_projections_per_class = []
            for q in self._classes:
                class_q_projections = projections[y == q]
                mean_projections_per_class.append(np.median(class_q_projections))
            mean_projections_per_class = np.array(mean_projections_per_class)
            self.thresholds = (
                mean_projections_per_class[:-1] + mean_projections_per_class[1:]
            ) / 2.0
        print("Threshold method:", self.thresholds_method, "thresholds:", self.thresholds)
        return self

    def predict(self, X):
        dmatrix_test = xgb.DMatrix(X)
        if self.booster is None:
            raise ValueError("Booster must be fitted before perdict")
        projections = self.booster.predict(dmatrix_test)
        predicted_classes = np.zeros(len(projections), dtype=int)
        for i, p in enumerate(projections):
            predicted_classes[i] = np.sum(p > self.thresholds)
        return predicted_classes

    def plot_projections(self, X, y, subsample_size=100, savepath="projections.png"):
        # plot porjections colored by originial class

        sss = StratifiedShuffleSplit(n_splits=1, test_size=subsample_size, random_state=42)
        _, idx = next(sss.split(X, y))
        if hasattr(X, "iloc"):
            X = X.iloc[idx]
            y = y.iloc[idx]
        else:
            X = X[idx]
            y = y[idx]

        dmatrix_test = xgb.DMatrix(X)
        if self.booster is None:
            raise ValueError("Booster must be fitted before perdict")
        projections = self.booster.predict(dmatrix_test)
        plt.figure(figsize=(10, 6))
        # Define a custom color map for the classes

        custom_colors = get_color_palete_from_n_classes(self._n_classes)
        cmap = ListedColormap(custom_colors[: self._n_classes])

        plt.scatter(
            projections,
            np.zeros(len(projections)),
            c=y.astype(int),  # assign color per label
            cmap=cmap,
            alpha=0.5,
        )
        for t in self.thresholds:
            plt.axvline(
                x=t, color="salmon", alpha=0.5, linestyle="--"
            )  # , label=f"Threshold {t:.2f}")
        plt.title("Projections with Thresholds")

        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=r"$\mathcal{C}_" + str(label) + "$",
            )
            for label, color in zip(self._classes, custom_colors)
        ]
        plt.legend(handles=legend_handles, loc="upper right")

        plt.savefig(savepath)
        plt.show()


def make_custom_obj(Q, thresholds, class_weights):
    def logat_obj(preds, dtrain):
        y = dtrain.get_label().astype(int)

        grad = np.zeros(len(y))
        hess = np.zeros(len(y))
        for i, y_i in enumerate(y):
            for q in range(y_i):
                grad[i] -= 1.0 / (1 + np.exp(preds[i] - thresholds[q]))
                hess[i] += np.exp(preds[i] - thresholds[q]) / (
                    (1 + np.exp(preds[i] - thresholds[q])) ** 2
                )
            for q in range(y_i, Q - 1):
                grad[i] += 1.0 / (1 + np.exp(thresholds[q] - preds[i]))
                hess[i] += np.exp(thresholds[q] - preds[i]) / (
                    (1 + np.exp(thresholds[q] - preds[i])) ** 2
                )
            grad[i] *= class_weights[y_i]
            hess[i] *= class_weights[y_i]

        return grad, hess

    return logat_obj
