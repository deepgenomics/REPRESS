# Copyright (2025) Deep Genomics Incorporated All rights reserved - no unauthorized use or reproduction
# Licensed under CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Please otherwise contact legal@deepgenomics.com
from pathlib import Path
import matplotlib.pyplot as plt
from genome_kit import Genome, Interval

from repress.model_wrapper import get_repress_model 
from repress.plot_tracks import make_track_plot
from repress.ism import InsilicoSaturationMutagenesis
from repress.plot_sequence_with_scores import plot_seq_scores

if __name__ == "__main__":
    analysis_dir = Path(__file__).parent 
    model = get_repress_model()

    genome = Genome("gencode.v29")
    gene_name = "UTRN"
    gene = [x for x in genome.genes if x.name == gene_name][0]
    transcript = gene.transcripts[1]
    utr3 = transcript.utr3s[0]

    cell_lines = ["K562", "A549"]

    prediction = model.predict_interval(utr3, genome, transcript=transcript, cell_lines=cell_lines)[0]

    fig, ax = make_track_plot(utr3, prediction, gene, genome, figsize=(8, 3),
                            scale_marker=0.4, custom_transcript=transcript)
    save_path = analysis_dir / "plots/UTRN_track_plot.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")

    ism = InsilicoSaturationMutagenesis(predictor=model)
    ism_interval = Interval("chr6", "+", 144852566, 144852651, genome)
    # ISM for A549 predictions
    points = ism.run_ism_on_itv(ism_interval, genome, transcript=transcript)[:, :, 13]

    plt.figure(figsize=(10, 1))
    ax = plot_seq_scores(points)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    save_path = analysis_dir / "plots/repress_A549_sequence_attribution.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")
