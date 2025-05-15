# Copyright (2025) Deep Genomics Incorporated All rights reserved - no unauthorized use or reproduction
# Licensed under CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Please otherwise contact legal@deepgenomics.com
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import metrics

def get_boostrapped_roc(labels, predictions, iterations=1000):

    rng = np.random.default_rng()  # Random number generator
    np.random.seed(42)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for i in range(iterations):
        boot = rng.choice(labels.shape[0], labels.shape[0]).astype(np.int32)

        fpr_boot, tpr_boot, _ = metrics.roc_curve(labels[boot], predictions[boot])

        tpr_array = np.interp(mean_fpr, fpr_boot, tpr_boot)
        tpr_array[0] = 0.0
        tprs.append(tpr_array)

    median_tpr = np.median(tprs, axis=0)
    median_tpr[-1] = 1.0
    #Normal distribution
    lower = np.percentile(tprs, q=10, axis=0)
    upper = np.percentile(tprs, q=90, axis=0)

    return mean_fpr, median_tpr, lower, upper

if __name__ == "__main__":
    analysis_dir = Path(__file__).parent 

    df = pd.read_csv(analysis_dir / "data/model_predictions.csv")
    df = df.dropna()

    predictions = np.array(df["prediction"].tolist())
    labels = np.array(df["label"].tolist())

    fpr, tpr, lower, upper = get_boostrapped_roc(labels, predictions)
    auroc = metrics.roc_auc_score(labels, predictions)
    auroc = round(auroc, 2)
    print(auroc)

    plt.figure(figsize=(4, 4))
    plt.title("REPRESS Variant Analysis", fontweight="semibold", fontsize=10)
    plt.grid(visible=True, c="lightgray", linewidth=0.5, zorder=0)
    plt.plot([0, 1], [0, 1], c="lightgrey", linestyle="--")
    plt.plot(fpr, tpr, "#0B8182", label=f"REPRESS, AUC = {auroc}")
    plt.gca().add_patch(patches.Polygon(list(zip(np.concatenate([fpr, fpr[::-1]]),
                                                np.concatenate([lower, upper[::-1]]))),
                                                linewidth=1,
                                                facecolor="#0B8182",
                                                alpha=0.2))
    plt.legend(loc="lower right", prop={'size': 8})
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(axis='both', which='both', direction='out', width=1.5, labelsize=8)
    plt.xlabel("False Positive Rate", fontweight="semibold", fontsize=10)
    plt.ylabel("True Positive Rate", fontweight="semibold", fontsize=10)
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.0))
    save_path = analysis_dir / "plots/repress_variant_roc.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")
