import sys
import numpy as np

from helpers import create_csv_submission
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
from model_selection import grid_search, best_threshold
from preprocess import preprocess_data

SEED = 1
K_FOLD = 5
np.random.seed(SEED)

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")
x_test = np.load("data/x_test.npy")
# test ids
test_ids = np.genfromtxt(
    "data/dataset/x_test.csv", delimiter=",", skip_header=1, usecols=0, dtype=int
)

y = (y_train + 1) / 2  # labels -1/1 -> 0/1
tx, tx_test = preprocess_data(x_train, x_test)
w0 = np.zeros(tx.shape[1])
print(np.mean(y_train == 1))

# ----------------------------------------------------------------------------
# Model selection and hyperparameter grids
# ----------------------------------------------------------------------------
EXPERIMENTS = {
    "MSE GD": (
        mean_squared_error_gd,
        {"gamma": [0.001, 0.01, 0.05]},
        {"initial_w": w0, "max_iters": 300},
    ),
    "MSE SGD": (
        mean_squared_error_sgd,
        {"gamma": [1e-4, 1e-3, 5e-3]},
        {"initial_w": w0, "max_iters": 20000},
    ),
    "Least squares": (least_squares, {}, {}),
    "Ridge": (ridge_regression, {"lambda_": list(np.logspace(-8, -1, 8))}, {}),
    "Logistic": (
        logistic_regression,
        {"gamma": [0.05, 0.1, 0.5]},
        {"initial_w": w0, "max_iters": 500},
    ),
    "Reg. logistic": (
        reg_logistic_regression,
        {"lambda_": [1e-5, 1e-4, 1e-3, 1e-2], "gamma": [0.5]},
        {"initial_w": w0, "max_iters": 500},
    ),
}

# Final model without --cv, results from previous runs, can be changed if needed
FINAL_MODEL = "Reg. logistic"
FINAL_PARAMS = {"lambda_": 0.001, "gamma": 0.5}

# ----------------------------------------------------------------------------
# --cv, optional
# ----------------------------------------------------------------------------
if "--cv" in sys.argv:
    summary = {}
    for name, (fn, grid, fixed) in EXPERIMENTS.items():
        print(f"\n=== {name} ===")
        (params, res), _ = grid_search(y, tx, fn, grid, fixed, k_fold=K_FOLD, seed=SEED)
        summary[name] = (params, res)

    print(f"\n{'Modèle':<15}{'F1 (CV)':>18}{'Accuracy':>12}   Meilleurs params")
    for name, (params, res) in sorted(summary.items(), key=lambda kv: -kv[1][1]["f1"]):
        print(
            f"{name:<15}{res['f1']:>10.4f} ± {res['f1_std']:.4f}{res['acc']:>10.4f}   {params}"
        )

    FINAL_MODEL = max(summary, key=lambda n: summary[n][1]["f1"])
    FINAL_PARAMS = summary[FINAL_MODEL][0]

# ----------------------------------------------------------------------------
# Final model training and submission
# ----------------------------------------------------------------------------
print(f"\nModèle final : {FINAL_MODEL} {FINAL_PARAMS}")
fn, _, fixed = EXPERIMENTS[FINAL_MODEL]
w, _ = fn(y, tx, **fixed, **FINAL_PARAMS)

t, f1_train = best_threshold(y, tx @ w)
print(f"Seuil = {t:.4f} | F1 train = {f1_train:.4f}")

y_pred = np.where(tx_test @ w >= t, 1, -1)  # AIcrowd attend des labels -1/1
create_csv_submission(test_ids, y_pred, "submission.csv")
print(f"submission.csv écrit ({np.mean(y_pred == 1):.1%} de prédictions positives)")
