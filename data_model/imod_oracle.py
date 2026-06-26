import os
import numpy as np
import pandas as pd

FEATURE_COLS = ["Anneal Time", "Temperature", "R MAI", "R BAAc"]
QW_COLS = [f"QW{i}" for i in range(1, 13)] + ["QW99"]
DEFAULT_CSV = os.path.join(os.path.dirname(__file__), 'data_noBAI.csv')


def load_pool(csv_path=None, donor_qw='QW1', target_qw='QW99'):
    """
    Load and return pool data for the real-data BO runner.

    Returns
    -------
    X_pool         : (N, 4) float array — feature rows where both QWs are present
    y_donor_pool   : (N,)   float array — donor QW values at pool locations (for feasibility)
    y_target_pool  : (N,)   float array — target QW values at pool locations (ground-truth objective)
    X_donor_train  : (M, 4) float array — all rows with donor QW present (for constraint GP)
    y_donor_train  : (M,)   float array — corresponding donor QW values
    """
    path = csv_path or DEFAULT_CSV
    data = pd.read_csv(path)
    data = data[
        (data["R MAI"] >= 0) & (data["R MAI"] <= 2.5) &
        (data["R BAAc"] >= 0) & (data["R BAAc"] <= 2.5)
    ].dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Donor training set: all rows where donor QW is measured
    donor_mask = data[donor_qw].notna()
    X_donor_train = data.loc[donor_mask, FEATURE_COLS].values.astype(float)
    y_donor_train = data.loc[donor_mask, donor_qw].values.astype(float)

    # Pool: rows where BOTH QWs are measured (BO can query and observe true target value)
    both_mask = data[donor_qw].notna() & data[target_qw].notna()
    pool = data[both_mask].reset_index(drop=True)
    X_pool        = pool[FEATURE_COLS].values.astype(float)
    y_donor_pool  = pool[donor_qw].values.astype(float)
    y_target_pool = pool[target_qw].values.astype(float)

    return X_pool, y_donor_pool, y_target_pool, X_donor_train, y_donor_train
