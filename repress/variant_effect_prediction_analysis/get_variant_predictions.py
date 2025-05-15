# Copyright (2025) Deep Genomics Incorporated All rights reserved - no unauthorized use or reproduction
# Licensed under CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Please otherwise contact legal@deepgenomics.com
from pathlib import Path
import numpy as np
import pandas as pd

from tqdm import tqdm
from genome_kit import Genome, Variant, VariantGenome

from repress.model_wrapper import get_repress_model 
from repress.disjoint_intervals import DisjointIntervalsSequence

def score_variant(wt, mut, score, first_axis=0):

    parts = score.split('_')
    if len(parts) == 3:

        if parts[0] == 'mean':
            wt = np.mean(wt, axis=first_axis)
            mut = np.mean(mut, axis=first_axis)
        elif parts[0] == 'max':
            wt = np.max(wt, axis=first_axis)
            mut = np.max(mut, axis=first_axis)
        else:
            print('Wrong part', parts[1])
            exit()

        score = '_'.join(parts[1:])

    if score == 'sum_diff':
        ret = np.absolute(np.mean(wt) - np.mean(mut))
    elif score == 'max_diff':
        ret = np.absolute(np.max(wt) - np.max(mut))
    elif score == 'diff_max':
        ret = np.max(np.absolute(wt - mut))
    else:
        print('Wrong Score')
        exit()

    return ret

if __name__ == "__main__":
    analysis_dir = Path(__file__).parent 

    df = pd.read_csv(analysis_dir /  "data/variant_effect.csv")
    genome = Genome("gencode.v29")

    model = get_repress_model()
    prediction = []

    for i, row in tqdm(df.iterrows()):

        transcript = genome.transcripts[row["transcript_id"]]
        variant = Variant.from_string(row["variant"], genome)

        itv = variant.interval
        if transcript.interval.strand == '-':
            itv = itv.as_opposite_strand()

        varg = VariantGenome(genome, variant)
        disj_tx = DisjointIntervalsSequence([ex.interval for ex in transcript.exons], genome)
        disj_tx_mut = DisjointIntervalsSequence([ex.interval for ex in transcript.exons], varg)

        try:
            q_itv = disj_tx.lift_interval(itv).expand(20)
            q_itv_wt = q_itv.intersect(disj_tx.interval)
            q_itv_mut = q_itv.intersect(disj_tx_mut.interval)

        except ValueError:
            prediction.append(None)
            continue

        output = model.predict_interval([q_itv_wt, q_itv_mut], [genome, varg], 
                                        transcript=[transcript, transcript], exonic_interval=True)
        # subset to only miRNA predictions
        output = [x[:, :29] for x in output]
        
        var_score = score_variant(output[0], output[1], "mean_sum_diff")
        prediction.append(var_score)

    df["prediction"] = prediction
    save_path = analysis_dir / "data/model_predictions.csv"
    df.to_csv(analysis_dir / "data/model_predictions.csv", index=False)
    print(f"Saved predictions to {save_path}")
