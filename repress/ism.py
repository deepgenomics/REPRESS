# Copyright (2025) Deep Genomics Incorporated All rights reserved - no unauthorized use or reproduction
# Licensed under CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Please otherwise contact legal@deepgenomics.com

import logging

from tqdm import tqdm
import genome_kit as gk
import numpy as np

from repress.model_wrapper import encode_sequence

class InsilicoSaturationMutagenesis:
    """
    Class for running in-silico saturation mutagenesis on a given interval
    using a given predictor.

    Args:
        predictor (mb.Predictor): a Predictor object
        predictor_directory (str): the path to the directory containing the
                                      predictor

    Attributes:
        predictor (mb.Predictor): a Predictor object
        NUCLEOTIDE_COMPLEMENTS (Dict[str, str]): a dictionary mapping each
    """

    def __init__(self, predictor=None):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.predictor = predictor

    def run_ism_on_seq(
            self,
            seq: str,
            padding: bool = True,
            batch_size: int = 20,
            wild_type_only: bool = True,
            transform_deg: bool = True,
            deg_lastk: int = 10,
    ):
        mutations = ["A", "C", "G", "T"]
        base_prediction = self.predictor.predict_seq(seq, padding=padding)[0]
        if deg_lastk > 0 and transform_deg:
            degreadome_preds = base_prediction[:, -deg_lastk:]
            degreadome_preds = degreadome_preds ** (1.0 / 0.375)
            base_prediction[:, -deg_lastk:] = degreadome_preds

        base_prediction = np.mean(base_prediction, axis=0)
        points = np.zeros((len(seq), 4, base_prediction.shape[0]))
        seq_list, ind_list = [], []

        for i in range(len(seq)):

            for j, muts in enumerate(mutations):
                
                if seq[i] == muts:
                    for k in range(base_prediction.shape[0]):
                        points[i][j][k] = base_prediction[k]

                seq_arr = list(seq)
                seq_arr[i] = muts
                seq_list.append("".join(seq_arr))
                ind_list.append((i, j))

            if len(seq_list) >= batch_size or i == len(seq) - 1:
                predictions = self.predictor.predict_seq(seq_list, padding=padding)
                for j in range(len(predictions)):
                    
                    if deg_lastk > 0 and transform_deg:
                        degreadome_preds = predictions[j][:, -deg_lastk:]
                        degreadome_preds = degreadome_preds ** (1.0 / 0.375)
                        predictions[j][:, -deg_lastk:] = degreadome_preds

                    arr = np.mean(predictions[j], axis=0)

                    for k in range(arr.shape[0]):
                        points[ind_list[j][0]][ind_list[j][1]][k] = arr[k]
                seq_list = []
                ind_list = []

        input_seq = encode_sequence(seq)
        if wild_type_only:

            for i in range(base_prediction.shape[0]):
                arr = base_prediction[i] - points[:, :, i]
                arr = np.mean(arr, axis=1)

                arr = input_seq * np.expand_dims(arr, axis=-1)
                points[:, :, i] = arr

        return points

    def run_ism_on_itv(
        self,
        itv: gk.Interval,
        genome: gk.Genome,
        transcript: gk.Transcript = None,
        batch_size: int = 20,
        wild_type_only: bool = True,
        transform_deg: bool = True,
        deg_lastk: int = 10,
    ) -> np.array:
        """Runs in-silico saturation mutagenesis on the provided interval
        using the provided genome.

        Args:
            itv (gk.Interval): interval of interest for ISM
            genome (gk.Genome): the reference genome of <itv>
            output_node (str): the name of the output node for the predictor

        Returns:
            np.array: a (L,4, K) array, where L is the length of <itv>, that
                      contains ISM results, K is the total number of cell lines.
        """
        mutations = ["A", "C", "G", "T"]
        base_prediction = self.predictor.predict_interval(itv, genome, transcript=transcript)[0]
        if deg_lastk > 0 and transform_deg:
            degreadome_preds = base_prediction[:, -deg_lastk:]
            degreadome_preds = degreadome_preds ** (1.0 / 0.375)
            base_prediction[:, -deg_lastk:] = degreadome_preds

        base_prediction = np.mean(base_prediction, axis=0)
        points = np.zeros((len(itv), 4, base_prediction.shape[0]))
        cur_itv = itv.end5.expand(0, 1)
        genome_list, ind_list = [], []
        
        for i in tqdm(range(len(itv))):
            var_itv = gk.Interval(cur_itv.chromosome, "+", cur_itv.start, cur_itv.start + 1, genome)
            base_seq = genome.dna(var_itv)

            for j, muts in enumerate(mutations):
                
                if base_seq == muts:
                    for k in range(base_prediction.shape[0]):
                        points[i][j][k] = base_prediction[k]
                    
                variant = gk.Variant(var_itv.chromosome, var_itv.start, base_seq, muts, genome)
                varg = gk.VariantGenome(genome, variant)
                genome_list.append(varg)
                ind_list.append((i, j))

            cur_itv = cur_itv.shift(1)

            if len(genome_list) >= batch_size or i == len(itv) - 1:

                predictions = self.predictor.predict_interval([itv] * len(genome_list), genome_list, transcript=[transcript] * len(genome_list))
                for j in range(len(predictions)):
                    
                    if deg_lastk > 0 and transform_deg:
                        degreadome_preds = predictions[j][:, -deg_lastk:]
                        degreadome_preds = degreadome_preds ** (1.0 / 0.375)
                        predictions[j][:, -deg_lastk:] = degreadome_preds

                    arr = np.mean(predictions[j], axis=0)
                    
                    for k in range(arr.shape[0]):
                        points[ind_list[j][0]][ind_list[j][1]][k] = arr[k]
                genome_list = []
                ind_list = []

        input_seq = encode_sequence(genome.dna(itv))
        if wild_type_only:

            for i in range(base_prediction.shape[0]):
                arr = base_prediction[i] - points[:, :, i]
                arr = np.mean(arr, axis=1)

                arr = input_seq * np.expand_dims(arr, axis=-1)
                points[:, :, i] = arr

        return points
