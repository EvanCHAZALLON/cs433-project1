# CS-433 Project 1 – Heart disease prediction (BRFSS 2015)

Thomas Claux, Evan Chazallon, Vadim Paccaud

We predict whether a respondent of the 2015 BRFSS survey has been diagnosed with
coronary heart disease (MICHD) from 321 survey answers. Only 8.8% of the training
respondents are positive, so models are compared on F1 rather than accuracy.
Everything is written with NumPy only (matplotlib is only used in the notebook).

Best submission on AIcrowd: **F1 = 0.441, accuracy = 0.875**
(5-fold CV estimate: F1 = 0.4252, accuracy = 0.8710).

## Repository

- `implementations.py` – the six required methods (GD / SGD for MSE, least squares,
  ridge, logistic and regularized logistic regression).
- `preprocess.py` – data cleaning and feature processing (`preprocess_data`).
- `model_selection.py` – F1 score, threshold selection, k-fold cross-validation and
  grid search.
- `run.py` – trains the final model and writes `submission.csv`.
- `helpers.py` – loading / submission helpers given by the course.
- `exploration.ipynb` – our first exploration of the data (split 80/20, polynomial
  features with ridge regression).

## Data

Put `x_train.csv`, `y_train.csv` and `x_test.csv` in `data/dataset/`.
Reading the CSV files is slow, so our scripts load `.npy` copies. Create them once with:

```python
import numpy as np
from helpers import load_csv_data

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset")
np.save("data/x_train.npy", x_train)
np.save("data/x_test.npy", x_test)
np.save("data/y_train.npy", y_train)
```

`run.py` still reads `data/dataset/x_test.csv` for the test ids, and `preprocess.py`
reads the header of `data/dataset/x_train.csv` for the column names.

## How to run

```bash
python run.py
```

This reproduces exactly our AIcrowd submission (`submission.csv`). It takes a few
minutes on a laptop.

```bash
python run.py --cv
```

This also runs the 5-fold cross-validation grid search over the six methods
(Table I of the report), which is much slower. Note that with `--cv` the script then
trains the best model found by the grid (lambda = 1e-4) instead of the one we
submitted (lambda = 1e-3), so use `python run.py` without option to get our
submission.

## Pipeline

Preprocessing (`preprocess_data`), all statistics computed on the training set:

1. Drop 29 columns that describe the interview and not the person (state, dates,
   record ids, phone questions, sampling weights). The list is `DROP_NAMES`.
2. Decode the BRFSS special answers using the maximum of each column:
   7/9, 77/99, 777/999 ("don't know", "refused") become missing, and 88/888
   ("none") become 0.
3. Drop the columns with at least 75% missing values.
4. Replace the remaining missing values by the median of the column.
5. Drop constant columns, standardize, and add a bias column (177 features in total).

One-hot encoding of the columns with at most 10 distinct values is implemented
(`one_hot_enc=True`) but is not used in our submission.

Model: L2-regularized logistic regression, gradient descent from `w = 0`,
500 iterations, `gamma = 0.5`, `lambda_ = 1e-3`.

Decision threshold: with so few positives, a probability threshold of 0.5 almost
never predicts MICHD. We test 60 thresholds (quantiles 40% to 99.5% of the scores)
and keep the one with the best F1 on the training data. In cross-validation this is
done on the training folds only. The final threshold corresponds to a probability
of about 0.2, and 13.7% of the test set is predicted positive.

Cross-validation: 5 folds, random split with seed 1, mean and standard deviation of
F1 and accuracy over the validation folds.

## Requirements

Python 3 and NumPy (matplotlib for the notebook). No other library is used.
