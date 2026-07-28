# 🌳 Welcome to `XGBORD`

This repository provides the official implementation of `XGBORD`, a novel eXtreme Gradient Boosting approach designed for **ordinal classification** tasks. This repository contains the code accompanying the scientific article:

> **XGBORD: eXtreme Gradient Boosting for Ordinal Classification**

* 🚀 Built on top of **XGBoost**, using its low-level API with a custom ordinal logistic loss function.
* 🚀 Introduces a novel **threshold optimization strategy** that exploits the natural order of class labels via cumulative probabilities.
* 🚀 Designed for **reproducibility** — all experiments from the article can be replicated with the provided notebooks.

✨ `XGBORD` is a **top-performing method** on the [TOCUCO](https://github.com/ayrna/tocuco) benchmark, the standard repository for ordinal classification comparison.

---

## Installation

Install all dependencies with `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
pip install -e .
```

---

## Usage

```python
from models.xgbord import XGBORD

clf = XGBORD(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    thresholds_method="lat_med",
    balanced_class_weights=True,
)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

## Remaining baseline techniques code
You can get the code of the remaining sota baseline techniques on [TOCUCO](https://github.com/ayrna/tocuco) and [ADABORD](https://github.com/ayrna/adabord)

---

## 📚 Citation

If you use `XGBORD` in your research, we would appreciate a citation for the following work:

*citation not available yet*
