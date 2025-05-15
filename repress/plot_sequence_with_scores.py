# Copyright (2025) Deep Genomics Incorporated All rights reserved - no unauthorized use or reproduction
# Licensed under CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Please otherwise contact legal@deepgenomics.com

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import numpy.typing as npt

# Many thanks to the Borzoi authors:
# https://github.com/calico/borzoi/blob/864bbba4e9fca68a2b374c176815e387f0069181/examples/borzoi_helpers.py

def dna_letter_at(letter: str, x: float, y: float, yscale: float, ax: mpl.axes.Axes, alpha: float=1.0) -> None:
    """Given a letter and an axis, plots the nucleotide on the axis.

    Args:
        letter (str): The nucleotide to be plotted. Must be on of A,T,C,G
        x (float): x coordinate of location
        y (float): y coordinate of location
        yscale (float): The score of the nucleotide itself
        ax (mpl.axes.Axes): The axes to plot on
        alpha (float, optional): Opacity parameter. Defaults to 1.0.
    """

    fp = FontProperties(family="DejaVu Sans", weight="bold")

    globscale = 1.35

    LETTERS = {
        "T": TextPath((-0.305, 0), "T", size=1, prop=fp),
        "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
        "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
        "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
        "U": TextPath((-0.366, 0), "U", size=1, prop=fp),
    }

    COLOR_SCHEME = {
        "G": "orange",
        "A": "green",
        "C": "blue",
        "T": "red",
        "U": "red",
    }

    text = LETTERS[letter]

    chosen_color = COLOR_SCHEME[letter]

    t = (
        mpl.transforms.Affine2D().scale(1 * globscale, yscale * globscale)
        + mpl.transforms.Affine2D().translate(x, y)
        + ax.transData
    )
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)

    if ax != None:
        ax.add_artist(p)

def _get_normalization_constant(sequence_scores):
    # get maximum positive height
    pos = sequence_scores * (sequence_scores >= 0)
    if pos.shape[0] == 0:
        pos_max_height = 0
    else:
        pos_max_height = np.max(np.sum(pos, axis=1))
    # get maximum negative height
    neg = sequence_scores * (sequence_scores <= 0)
    if neg.shape[0] == 0:
        neg_max_height = 0
    else:
        neg_max_height = np.max(np.sum(np.abs(neg), axis=1))

    return pos_max_height, neg_max_height


def plot_seq_scores(
    sequence_scores: npt.NDArray, use_u: bool = False
) -> mpl.axes.Axes:
    """Sequence score plotting function. Nucleotide order is ATCG. Example usage:
    import numpy as np
    import matplotlib.pyplot as plt

    array = np.random.rand(5,4) * 2 -1
    ax = plot_seq_scores(array)
    plt.show()

    Args:
        sequence_scores (npt.NDArray): Lx4 array of sequence scores.
        use_u (bool): whether to use U instead of T
    """
    if sequence_scores.shape[1] != 4:
        raise ValueError(f"Expected an Lx4 array, got Lx{sequence_scores.shape[1]}")

    max_pos, max_neg = _get_normalization_constant(sequence_scores)
    norm = max(max_pos, max_neg)
    normalized_scores = sequence_scores/norm

    if use_u:
        LETTERS = ['A', 'C', 'G', 'U']
    else:
        LETTERS = ['A', 'C', 'G', 'T']

    ax = plt.gca()

    for i in range(0, normalized_scores.shape[0]):
        negative_score_so_far = 0
        # plot negative scores first
        negative_scores = [(n, s) for n, s in zip(LETTERS,list(normalized_scores[i])) if s < 0]
        negative_scores = sorted(negative_scores, key=lambda pair: -pair[1])
        for letter, score in negative_scores:
            score = normalized_scores[i, LETTERS.index(letter)]
            dna_letter_at(letter, i + 0.5, negative_score_so_far, score, ax)
            negative_score_so_far += score

        # plot positive scores after
        positive_score_so_far = 0
        positive_scores = [(n, s) for n, s in zip(LETTERS,list(normalized_scores[i])) if s >= 0]
        positive_scores = sorted(positive_scores, key=lambda pair: pair[1])
        for letter, score in positive_scores:
            score = normalized_scores[i, LETTERS.index(letter)]
            dna_letter_at(letter, i + 0.5, positive_score_so_far, score, ax)
            positive_score_so_far += score


    plt.sca(ax)
    plt.xticks([], [])
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    plt.xlim((0, normalized_scores.shape[0]))

    plt.ylim(
    -max_neg/norm, max_pos/norm
    )

    plt.axhline(y=0.0, color="black", linestyle="-", linewidth=1)

    plt.tight_layout()

    return ax
