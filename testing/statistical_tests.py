import pandas as pd
from scipy.stats import ttest_rel
import numpy as np

def run_ttest(file_path):
    df = pd.read_csv(file_path)

    ara = df[df["model_config"] == "full_ara"]
    rag = df[df["model_config"] == "baseline_rag"]

    ara_acc = ara.groupby("run_id").apply(
    lambda x: (x["ground_truth"] == x["predicted_label"]).mean()
    )

    rag_acc = rag.groupby("run_id").apply(
        lambda x: (x["ground_truth"] == x["predicted_label"]).mean()
    )

    common_runs = set(ara_acc.index).intersection(set(rag_acc.index))

    ara_acc = ara_acc.loc[list(common_runs)]
    rag_acc = rag_acc.loc[list(common_runs)]
    t_stat, p_value = ttest_rel(ara_acc, rag_acc)

    print("ARA Mean ± Std:", np.mean(ara_acc), "±", np.std(ara_acc))
    print("RAG Mean ± Std:", np.mean(rag_acc), "±", np.std(rag_acc))
    print("p-value:", p_value)