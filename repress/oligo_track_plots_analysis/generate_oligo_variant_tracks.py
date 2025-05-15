# Copyright (2025) Deep Genomics Incorporated All rights reserved - no unauthorized use or reproduction
# Licensed under CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Please otherwise contact legal@deepgenomics.com
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from genome_kit import Genome, Variant, VariantGenome, Interval

from repress.model_wrapper import get_repress_model 
from repress.plot_tracks import make_track_plot

def create_oligo_genome(genome, oligo_interval):

    oligo_interval = oligo_interval.as_positive_strand()
    seq = genome.dna(oligo_interval)
    replace_seq = "N" * len(seq)

    variant = Variant(oligo_interval.chromosome, oligo_interval.start, seq, replace_seq, genome)
    mutant_genome = VariantGenome(genome, variant)

    return mutant_genome

if __name__ == "__main__":
    analysis_dir = Path(__file__).parent 

    model = get_repress_model()

    genome = Genome("gencode.v29")
    option = "variant" # "variant" or "oligo"

    if option == "variant":
        gene_name = "COL4A1"
        gene = [x for x in genome.genes if x.name == gene_name][0]
        transcript = gene.transcripts[0]
        utr3 = transcript.utr3s[0]
        cell_line = "A2780"
        ylabels = ["Wild type prediction", "Variant prediction"]

        variant = Variant(utr3.chromosome, 110150330, "C", "A", genome)
        mutant_genome = VariantGenome(genome, variant)
        vertical_marker = 110150330

    elif option == "oligo":
        gene_name = "UTRN"
        gene = [x for x in genome.genes if x.name == gene_name][0]
        transcript = gene.transcripts[1]
        utr3 = transcript.utr3s[0]
        cell_line = "HEK293"

        oligo_interval = Interval("chr6", "+", 144852583, 144852609, genome)
        mutant_genome = create_oligo_genome(genome, oligo_interval)
        vertical_marker = (144852583 + 144852609) // 2
        ylabels = ["Wild type prediction", "Oligo prediction"]

    else:
        raise ValueError("Invalid option")


    prediction = model.predict_interval([utr3, utr3], [genome, mutant_genome],
                            transcript=[transcript, transcript], cell_lines=cell_line)
    prediction = np.concatenate(prediction, axis=-1)

    fig, ax = make_track_plot(utr3, prediction, gene, genome, figsize=(8, 3),
                            scale_marker=0.4, custom_transcript=transcript,
                            vertical_markers=[vertical_marker], vertical_markers_colors=["red"],
                            ylabels=ylabels, labelpad=40, absolute_scale=True)
    save_path = analysis_dir / f"plots/repress_{option}_track_plot.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")
