# Copyright (2025) Deep Genomics Incorporated All rights reserved - no unauthorized use or reproduction
# Licensed under CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Please otherwise contact legal@deepgenomics.com
from pathlib import Path
import os
import zipfile
from typing import List, Optional, Union

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input
from genome_kit import Interval, Genome, Transcript

from repress.disjoint_intervals import DisjointIntervalsSequence

def load_zip(path):
    """
    Given the path to a zip file returns the path to the extracted directory.
    Bypass extraction if folder already exists
    """
    file_directory = path + "_unzip"
    if not os.path.exists(file_directory):
        with zipfile.ZipFile(path, "r") as zipobj:
            zipobj.extractall(file_directory)
    return file_directory

def encode_sequence(
    seq: str,
    order: str = "ACGT",
) -> np.ndarray:
    """Encode sequence into numpy array.

    Args:
        seq:
            Sequence to be encoded, upper and lower case characters are
            being treated the same.
        order:
            Permutation of string ``ACGT``, the order determines which column we assign
            to each character in the 1-hot encoding
            For example, if order = 'ACGT',
            A gets column 0, C gets 1, G gets 2 and T/U gets 3
    Returns:
        Encoded sequence as a numpy array with shape (sequence_length, 4)
    """
    assert set(list(order)) == set(list("ACGT"))

    encoding = np.zeros([90, 4], np.float32)
    encoding[[ord(base) for base in order]] = np.eye(4)

    valid_strs = b"ACGTN"
    y = seq.encode("ascii").upper()
    assert all(x in valid_strs for x in y)
    return encoding[memoryview(y)]

class REPRESS:

    def __init__(self, path, cell_type_csv, models=["fold1", "fold2", "fold3", "fold4"],
                 context_size=12500):
        
        self.context_size = context_size

        model_input = Input(shape=(None, 4), name="model_input")
        ensemble_outputs = []

        for i, model in enumerate(models):

            model_path = f"{path}/{model}.zip"
            model_path = load_zip(model_path)

            mod = tf.keras.models.load_model(
                model_path, compile=False, custom_objects={"tf": tf}
            )

            mod_bld = mod(model_input)
            mod_bld = keras.models.Model(inputs=model_input, outputs=mod_bld)
            mod_bld.layers[1]._name = "functional_" + str(i)

            ensemble_outputs.append(mod_bld)

        if len(models) > 1:
            model_output = [model.output for model in ensemble_outputs]
        else:
            model_output = [ensemble_outputs[0].output]

        self.model = keras.models.Model(inputs=model_input, outputs=model_output)
        self.model.train = False

        self.cell_types_index = {}
        cell_type_df = pd.read_csv(cell_type_csv)
        for i, row in cell_type_df.iterrows():
            self.cell_types_index[row["cell_type"]] = i


    def _predict(self, x: np.ndarray):
        """
        Parameters
        ----------
        x
                Numpy array of shape (n, l, 4)

        Returns
        -------
                Numpy array of shape (n, l, 6) corresponding to the AGO2 binding
                probabilities
                at each base pair. Here 6 is the total number of cell lines.
        """
        assert len(x.shape) == 3
        assert x.shape[2] == 4

        return self.model.predict(x, verbose=0)
    
    def _predict_seq(self, seq: List[str], transform_deg: bool = True) -> np.ndarray:
        
        seq_len = [len(seq[i]) - self.context_size for i in range(len(seq))]
        assert all(seq_len) > 0, "Sequence length should be greater than 0"
        assert len(set(seq_len)) == 1, "All sequence lengths must be same"

        x = []
        for i in range(len(seq)):
            x.append(encode_sequence(seq[i]))
        
        x = np.array(x)
        output = self._predict(x)
        output = np.array(output)

        if len(output.shape) == 3:
            output = np.expand_dims(output, axis=0)

        output = np.mean(output, axis=0)
        if transform_deg:
            # Invert squashed transformation
            output[..., -10:] = output[..., -10:] ** (1.0 / 0.375)

        return output
    
    def _subset_to_cell_lines(
        self, y: List[np.ndarray], cell_lines: Union[str, List[str]]
    ) -> List[np.ndarray]:
        if isinstance(cell_lines, str):
            cell_lines = [cell_lines]

        new_y = []
        for i in range(len(y)):
            return_vals = []
            for line in cell_lines:
                assert (
                    line in self.cell_types_index.keys()
                ), "Not a valid cell type"
                idx = self.cell_types_index[line]
                return_vals.append(y[i][:, idx])

            new_y.append(np.array(return_vals).transpose())

        return new_y
    
    def _correct_lengths(
        self, output: np.ndarray, seq_len: List[int]
    ) -> List[np.ndarray]:
        """Crops the added padding from output to return sequences to 
        original lengths"""
        y = []
        for i in range(len(seq_len)):
            y.append(output[i, : seq_len[i], :])

        return y
    
    def _pad_seqs(
        self, seqs: List[str]
    ) -> str:
        for i in range(len(seqs)):
            seqs[i] = (
                        "N" * (self.context_size // 2)
                        + seqs[i]
                        + "N" * (self.context_size // 2)
                    )
        return seqs
    
    def _get_seq_lens(self, seqs: List[str]):
        return [len(seqs[i]) - self.context_size for i in range(len(seqs))]
    
    def _add_len(
        self, seqs: List[str]
    ) -> str:
        seq_len = [len(seqs[i]) - self.context_size for i in range(len(seqs))]
        max_len = max(seq_len)

        for i in range(len(seqs)):
            add_len = max_len - seq_len[i]
            if add_len > 0:
                seqs[i] = seqs[i] + "N" * add_len
        return seqs

    def predict_seq(
        self,
        seq: Union[str, List[str]],
        padding: bool = True,
        cell_lines: Optional[List[str]] = None,
        transform_deg: bool = True,
    ) -> List[np.ndarray]:
        """Predicts AGO2 binding probabilites on a sequence

        Parameters
        ----------
        seq
                Input sequence / sequences of nucleotides
        padding
                Choose to pad the sequence with N's or no padding
        cell_lines
                A list of length k corresponding to the cell lines to be
                included in the output
                if None, it will return all cell lines by default.
        normalize_by
                Normalize the cell type outputs by - precision,
                selectivity or percentile.
        Returns
        -------
            Returns a list of numpy arrays  each of size (l_i, 1)
        corresponding to the ith input sequence given the ith sequence has
        a length l_i if cell_lines is a `str` or (l, k) if it is a `list`.
        The numpy array corresponds the AGO2 binding probabilities at each
        nucleotide of the sequence.
        """
        if not isinstance(seq, list):
            seq = [seq]

        if padding:
            seq = self._pad_seqs(seq)

        seq_len = self._get_seq_lens(seq)
        seq = self._add_len(seq)

        y = self._predict_seq(seq, transform_deg=transform_deg)
        y = self._correct_lengths(y, seq_len)

        if cell_lines is None:
            return y
        else:
            return self._subset_to_cell_lines(y, cell_lines)
        
    def predict_interval(
        self,
        interval: Union[Interval, List[Interval], DisjointIntervalsSequence],
        genome: Union[Genome, List[Genome]],
        pad_with_real_sequence: bool = True,
        transcript: Optional[Union[Transcript, List[Transcript]]] = None,
        cell_lines: Optional[List[str]] = None,
        exonic_interval: bool = False,
        transform_deg: bool = True,
    ) -> List[np.ndarray]:
        """Predicts AGO2 binding probabilities on a GenomeKit Interval

        Parameters
        ----------
        interval
                Interval / Intervals of interest.
        genome
                Genome of the species of interest.
        pad_with_real_sequence
                This decides whether to pad using actual genomic sequence or with N's.
                By default it will pad with an actual genomic sequence.
        transcript
                Provides the transcripts that the intervals belongs to.
                This is can be done for better padding.
                To make sure regions outside the gene are not used for
                generating the prediction.
        cell_lines
                A list of length k corresponding to the cell lines to be
                included in the output
                if None, it will return all cell lines by default.
        exonic_interval
                If the interval is already known to be disjoint, set this to True
                to avoid lifting the interval into disjoint coordinates.

        Returns
        -------
            Returns a list of numpy arrays  each of size (l_i, 1)
        corresponding to the ith interval given the ith sequence has
        a length l_i if cell_lines is a `str` or (l, k) if it is a `list`.
        The numpy array corresponds the AGO2 binding probabilities at each
        nucleotide of the sequence.
        """
        if not isinstance(interval, list):
            interval = [interval]

        if not isinstance(transcript, list) and transcript is not None:
            transcript = [transcript]

        if not isinstance(genome, list):
            batch_size = len(interval)
            genome = [genome] * batch_size

        assert len(interval) == len(
            genome
        ), "Must have same number of intervals and genomes"

        if transcript is not None:
            assert len(interval) == len(
                transcript
            ), "Must have same number of intervals and transcripts"

        seq_len = [len(interval[i]) for i in range(len(interval))]
        assert all(seq_len) > 0, "Intervals must all be non-zero length"

        seqs = []
        for i in range(len(interval)):

            itv = interval[i]

            if transcript is not None:
                tx = transcript[i]

            #Gathering exonic sequence if transcript is provided
            if transcript is not None:
                disj_tx = DisjointIntervalsSequence(
                    [ex.interval for ex in tx.exons], genome[i]
                )
                #If the input interval is already known to be exonic
                if exonic_interval == True:
                    itv = interval[i]
                #Else we have to lift it into exonic coordinates
                else:
                    itv = disj_tx.lift_interval(interval[i])
                tx = disj_tx
                use_genome = disj_tx
            else:
                use_genome = genome[i]

            #Padding with sequence
            if not pad_with_real_sequence:
                seq = use_genome.dna(itv)
                seqs.append(seq)
            else:
                if transcript is None:
                    seq = use_genome.dna(
                        itv.expand(
                            self.context_size // 2, self.context_size // 2
                        )
                    )
                else:
                    tx_itv = tx.interval
                    left_exp = itv.end5.expand(self.context_size // 2, 0)
                    right_exp = itv.end3.expand(0, self.context_size // 2)

                    left_overlap = left_exp.intersect(tx_itv)
                    right_overlap = right_exp.intersect(tx_itv)
                    len_left, len_right = (
                        self.context_size // 2,
                        self.context_size // 2,
                    )
                    left_seq, right_seq = "", ""
                    if left_overlap is not None:
                        len_left = self.context_size // 2 - len(left_overlap)
                        left_seq = use_genome.dna(left_overlap)
                    if right_overlap is not None:
                        len_right = self.context_size // 2 - len(right_overlap)
                        right_seq = use_genome.dna(right_overlap)
                    seq = (
                        "N" * len_left
                        + left_seq
                        + use_genome.dna(itv)
                        + right_seq
                        + "N" * len_right
                    )

                seqs.append(seq)

        if not pad_with_real_sequence:
            y = self.predict_seq(seqs, padding=True, transform_deg=transform_deg)
        else:
            y = self.predict_seq(seqs, padding=False, transform_deg=transform_deg)

        if cell_lines is None:
            return y
        else:
            return self._subset_to_cell_lines(y, cell_lines)

def get_repress_model():
    repress_dir = Path(__file__).parent
    model = REPRESS(str(repress_dir / "repress_model"), str(repress_dir / "cell_line_csv.csv"))
    return model
