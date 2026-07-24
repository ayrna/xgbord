import numpy as np


def stable_logistic_function(t):
    """
    Convert projections to probabilities.
    """
    idx = t > 0
    out = np.zeros_like(t)
    out[idx] = 1.0 / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1.0 + exp_t)
    return out


def stable_inverse_logistic_function(p):
    """
    Convert probabilities to projections.
    """
    idx = p > 0.5
    out = np.zeros_like(p)
    out[idx] = -np.log((1.0 / p[idx]) - 1.0)
    out[~idx] = np.log(p[~idx] / (1.0 - p[~idx]))
    return out


def get_color_palete_from_n_classes(n_classes):
    custom_colors = [
        "#000000",  # 0
        "#f94144",  # 1
        "#f3722c",  # 2
        "#f8961e",  # 3
        "#f9c74f",  # 4
        "#90be6d",  # 5
        "#43aa8b",  # 6
        "#3DCB2E",  # 7
        "#0AFF44",  # 8
        "#29f8ff",  # 9
    ]
    if n_classes == 3:
        return [custom_colors[1], custom_colors[4], custom_colors[7]]
    elif n_classes == 4:
        return [custom_colors[1], custom_colors[3], custom_colors[5], custom_colors[7]]
    elif n_classes == 5:
        return [
            custom_colors[1],
            custom_colors[3],
            custom_colors[5],
            custom_colors[7],
            custom_colors[9],
        ]
    elif n_classes == 6:
        return [
            custom_colors[1],
            custom_colors[2],
            custom_colors[4],
            custom_colors[6],
            custom_colors[7],
            custom_colors[9],
        ]
    elif n_classes == 7:
        return [
            custom_colors[1],
            custom_colors[2],
            custom_colors[4],
            custom_colors[5],
            custom_colors[6],
            custom_colors[7],
            custom_colors[9],
        ]
    elif n_classes == 8:
        return [
            custom_colors[1],
            custom_colors[2],
            custom_colors[3],
            custom_colors[4],
            custom_colors[5],
            custom_colors[6],
            custom_colors[7],
            custom_colors[9],
        ]
    elif n_classes == 9:
        return [
            custom_colors[1],
            custom_colors[2],
            custom_colors[3],
            custom_colors[4],
            custom_colors[5],
            custom_colors[6],
            custom_colors[7],
            custom_colors[8],
            custom_colors[9],
        ]
    elif n_classes == 10:
        return custom_colors


# testing
if __name__ == "__main__":
    projections = np.array([-1000.0, -10.0, 0.5, 2.3, 1000000.0])
    probas = [0.000001, 0.45, 0.87, 0.999999]

    print("projections:")
    for p in projections:
        print(f"  {p:12.6f} -> {stable_logistic_function(np.array([p]))[0]:.6f}")
    print("probas:")
    for p in probas:
        print(f"  {p:12.6f} -> {stable_inverse_logistic_function(np.array([p]))[0]:.6f}")

import numpy as np
from sklearn.metrics import confusion_matrix


def amae(y_true: np.ndarray, y_pred: np.ndarray):
    """Copied from dlordinal package: https://github.com/ayrna/dlordinal"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_class = cm.shape[0]
    costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costs = np.abs(costs - np.transpose(costs))
    errors = costs * cm

    # Remove rows with all zeros in the confusion matrix
    non_zero_cm_rows = ~np.all(cm == 0, axis=1)
    errors = errors[non_zero_cm_rows]
    cm = cm[non_zero_cm_rows]

    per_class_maes = np.sum(errors, axis=1) / np.sum(cm, axis=1).astype("double")
    return np.mean(per_class_maes)
