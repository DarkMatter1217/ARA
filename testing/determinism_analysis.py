import numpy as np
import pandas as pd

def compute_variance(file_path):
    df = pd.read_csv(file_path)

    grouped = df.groupby(["claim_id"])

    variances = []

    for _, group in grouped:
        numeric = group["confidence"].values
        if len(numeric) > 1:
            variances.append(np.var(numeric))
    if not variances:
        print("Not enough repeated runs to compute variance.")
        return
    print("Mean Variance:", np.mean(variances))
