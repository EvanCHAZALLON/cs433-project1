import numpy as np


def clean_special_codes(x):
    """Remplace les codes 'ne sait pas / refus' par NaN et 'aucun' par 0, selon le max de la colonne."""
    x = x.copy()
    for j in range(x.shape[1]):
        col = x[:, j]
        m = np.nanmax(col) if not np.all(np.isnan(col)) else 0
        if m == 9:
            col[np.isin(col, [7, 9])] = np.nan
        elif m == 99:
            col[np.isin(col, [77, 99])] = np.nan
            col[col == 88] = 0
        elif m == 999:
            col[np.isin(col, [777, 999])] = np.nan
            col[col == 888] = 0
    return x


def one_hot(x_tr, x_te, max_cat=10):
    """Sépare les colonnes catégorielles (<= max_cat valeurs distinctes) en one-hot.
    Les catégories sont apprises sur le TRAIN uniquement."""
    cont_tr, cont_te, oh_tr, oh_te = [], [], [], []
    for j in range(x_tr.shape[1]):
        vals = np.unique(x_tr[~np.isnan(x_tr[:, j]), j])
        if len(vals) <= max_cat:
            for v in vals[
                1:
            ]:  # on saute une catégorie (évite la colinéarité avec le biais)
                oh_tr.append(x_tr[:, j] == v)
                oh_te.append(x_te[:, j] == v)
            oh_tr.append(np.isnan(x_tr[:, j]))  # NaN = sa propre catégorie
            oh_te.append(np.isnan(x_te[:, j]))
        else:
            cont_tr.append(x_tr[:, j])
            cont_te.append(x_te[:, j])
    return (
        np.column_stack(cont_tr),
        np.column_stack(cont_te),
        np.column_stack(oh_tr).astype(float),
        np.column_stack(oh_te).astype(float),
    )


DROP_NAMES = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "CTELENUM",
    "PVTRESD1",
    "COLGHOUS",
    "STATERES",
    "CELLFON3",
    "LADULT",
    "CTELNUM1",
    "CELLFON2",
    "CADULT",
    "PVTRESD2",
    "CCLGHOUS",
    "CSTATE",
    "LANDLINE",
    "QSTVER",
    "QSTLANG",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_LLCPWT",
]


def get_feature_names(csv_path="data/dataset/x_train.csv", n_cols=None):
    """Lit les noms de colonnes dans l'en-tête du CSV.
    Si l'en-tête a une colonne de plus que les données (la colonne 'Id'), on l'enlève.
    """
    with open(csv_path) as f:
        names = f.readline().strip().split(",")
    if n_cols is not None and len(names) == n_cols + 1:
        names = names[1:]
    if n_cols is not None:
        assert (
            len(names) == n_cols
        ), f"{len(names)} noms pour {n_cols} colonnes : vérifier l'alignement"
    return names


def names_to_indices(names, to_drop):
    """Convertit une liste de noms en indices (ignore les noms absents, avec un avertissement)."""
    missing = [c for c in to_drop if c not in names]
    if missing:
        print(f"Colonnes introuvables (ignorées) : {missing}")
    return [names.index(c) for c in to_drop if c in names]


def drop_columns(x_tr, x_te, idx):
    """Supprime les colonnes d'indices idx dans x_tr et x_te."""
    keep = np.ones(x_tr.shape[1], dtype=bool)
    keep[idx] = False
    return x_tr[:, keep], x_te[:, keep]


def preprocess_data(
    x_tr, x_te, nan_value=0.75, clean_codes=True, drop_col=True, one_hot_enc=False
):

    if drop_col:
        names = get_feature_names(n_cols=x_tr.shape[1])
        idx = sorted(set(names_to_indices(names, DROP_NAMES)))
        x_tr, x_te = drop_columns(x_tr, x_te, idx)
        print(f"{len(idx)} colonnes supprimées")

    if clean_codes:
        x_tr = clean_special_codes(x_tr)
        x_te = clean_special_codes(x_te)

    ## Remove features with too many NaN values
    keep = np.isnan(x_tr).mean(axis=0) < nan_value
    x_tr = x_tr[:, keep]
    x_te = x_te[:, keep]

    ## One-hot encode categorical features (BEFORE median imputation: NaN is its own category)
    if one_hot_enc:
        x_tr, x_te, oh_tr, oh_te = one_hot(x_tr, x_te)
        keep = oh_tr.std(axis=0) > 0  # remove one-hot features with 0 variance
        oh_tr, oh_te = oh_tr[:, keep], oh_te[:, keep]
    else:
        oh_tr, oh_te = np.empty((len(x_tr), 0)), np.empty((len(x_te), 0))

    ## Replace remaining NaN values with median of the feature
    med = np.nanmedian(x_tr, axis=0)
    x_tr = np.where(np.isnan(x_tr), med, x_tr)
    x_te = np.where(np.isnan(x_te), med, x_te)

    ## Delete features with zero variance (division by 0, gives useless values)
    std = x_tr.std(axis=0)
    keep = std > 0
    x_tr = x_tr[:, keep]
    x_te = x_te[:, keep]
    std = std[keep]

    ## Standardize features
    mean = x_tr.mean(axis=0)
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    ## Add bias column (column of ones) to keep w0, the model is not forced to go through the origin
    x_tr = np.c_[np.ones(len(x_tr)), x_tr]
    x_te = np.c_[np.ones(len(x_te)), x_te]

    return x_tr, x_te
