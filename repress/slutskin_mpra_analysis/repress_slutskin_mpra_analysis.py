# Copyright (2025) Deep Genomics Incorporated All rights reserved - no unauthorized use or reproduction
# Licensed under CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Please otherwise contact legal@deepgenomics.com
from pathlib import Path
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, gaussian_kde

from repress.model_wrapper import get_repress_model 

if __name__ == "__main__":

    analysis_dir = Path(__file__).parent 
    df = pd.read_csv(analysis_dir / "data/slutskin_mpra_dataset_subset.csv")
    model = get_repress_model()
    # pick from gp_k562, tr_k562, tr_hek293, tr_hepg2, tr_mcf7
    slutskin_cell_line = "gp_k562"
    cell_line = "K562"

    repress_cell_lines = ["K562", "HEK293", "MCF7", "HEK293T"]
    assert cell_line in repress_cell_lines

    scalar_output = {x: [] for x in repress_cell_lines}
    full_output = []

    batch_size = 20
    n_batches = math.ceil(len(df) / batch_size)

    for i in tqdm(range(n_batches)):

        temp_df = df.iloc[i * batch_size : (i + 1) * batch_size]
        assert len(temp_df) > 0
        seqs = temp_df["seq"].tolist()

        output = np.array(model.predict_seq(seqs, cell_lines=repress_cell_lines))
        full_output.append(output)
        for j in range(len(seqs)):
            for k in range(len(repress_cell_lines)):
                scalar_output[repress_cell_lines[k]].append(output[j, :, k].mean()) #(batch_size, len, n_cell_lines)

    predictions = np.array(scalar_output[cell_line])
    labels = np.array(df[slutskin_cell_line].tolist())[:predictions.shape[0]]

    print(predictions.shape, labels.shape)

    xy = np.vstack([predictions, labels])
    z = gaussian_kde(xy)(xy)

    sp, _ = spearmanr(predictions, labels)
    pr, _ = pearsonr(predictions, labels)
    pr = round(pr, 2)

    plt.figure(figsize=(3, 2))
    plt.axhline(0.0, linestyle='--', color='grey', linewidth=0.5)
    plt.title(f"{slutskin_cell_line} : Pearson = {pr}", fontsize=6)
    plt.scatter(predictions, labels, s=0.2, c=z, cmap=sns.color_palette("mako", as_cmap=True))
    plt.xlabel(f"REPRESS {slutskin_cell_line} Prediction", fontweight="semibold", fontsize=6)
    plt.ylabel("MPRA log2 Fold Change", fontweight="semibold", fontsize=6)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.tick_params(axis='both', which='both', direction='out', width=1.5, labelsize=8)
    save_path = analysis_dir / "plots/repress_slutskin_analysis.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")
