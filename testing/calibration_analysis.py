import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calibration_curve(file_path):
    df = pd.read_csv(file_path)
    df = df[df["confidence"].notna()]

    bins = np.linspace(0, 1, 6)
    df["bin"] = pd.cut(df["confidence"], bins)

    grouped = df.groupby("bin")

    accuracies = grouped.apply(
        lambda x: (x["ground_truth"] == x["predicted_label"]).mean()
    )

    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

    plt.plot(bin_centers, accuracies.values)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.show()
    
    
    
    