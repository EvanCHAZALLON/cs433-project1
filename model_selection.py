import itertools
import numpy as np


def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def best_threshold(y_true, scores, n_grid=60):

    candidates = np.unique(
        np.quantile(scores, np.linspace(0.40, 0.995, n_grid))
    )  ## Using np.quantile -> proportion, (60% of positives, 0.05 % of positives...)
    f1s = [
        f1_score(y_true, (scores >= t).astype(int)) for t in candidates
    ]  ## Testing f1 score for each threshold
    i = int(np.argmax(f1s))
    return candidates[i], f1s[i]


def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validate(y, tx, train_fn, k_fold=5, seed=1, verbose=False):

    folds = build_k_indices(y, k_fold, seed)
    f1s, accs = [], []
    for k in range(k_fold):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(k_fold) if j != k])

        w = train_fn(y[tr], tx[tr])
        t, _ = best_threshold(y[tr], tx[tr] @ w)
        pred = (tx[va] @ w >= t).astype(int)

        f1s.append(f1_score(y[va], pred))
        accs.append(accuracy(y[va], pred))
        if verbose:
            print(f"  fold {k + 1}/{k_fold}: F1={f1s[-1]:.4f}  acc={accs[-1]:.4f}")

    return {
        "f1": np.mean(f1s),
        "f1_std": np.std(f1s),
        "acc": np.mean(accs),
        "acc_std": np.std(accs),
    }


def grid_search(
    y, tx, model_fn, param_grid, fixed=None, k_fold=5, seed=1, verbose=True
):

    fixed = fixed or {}
    names = list(param_grid)
    results = []
    for values in itertools.product(*(param_grid[n] for n in names)):
        params = dict(zip(names, values))

        def train_fn(y_, tx_, p=params):
            return model_fn(y_, tx_, **fixed, **p)[0]

        res = cross_validate(y, tx, train_fn, k_fold, seed)
        results.append((params, res))
        if verbose:
            print(
                f"  {params} -> F1 {res['f1']:.4f} ± {res['f1_std']:.4f} | acc {res['acc']:.4f}"
            )

    best = max(results, key=lambda r: r[1]["f1"])
    return best, results
