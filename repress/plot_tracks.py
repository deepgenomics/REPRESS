# Copyright (2025) Deep Genomics Incorporated All rights reserved - no unauthorized use or reproduction
# Licensed under CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Please otherwise contact legal@deepgenomics.com

import numpy as np
import matplotlib.pyplot as plt
from genome_kit import Genome, Interval
from matplotlib.patches import Rectangle, RegularPolygon

from repress.disjoint_intervals import DisjointIntervalsSequence

def setup_track_axes(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([])
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_tick_params(labelsize=8)
    tick_labels = ax.get_yticklabels()
    for label in tick_labels:
        label.set_verticalalignment('top')

def setup_genome_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def add_transcript_rectangles(ax, interval, transcript, h_start, num_arrows=7, scale_marker=1.0, variant_location=None):
    
    tx_interval = transcript.interval
    ax.plot(np.arange(tx_interval.start, tx_interval.end), np.array([h_start + 0.15] * len(tx_interval)),
            color="grey", zorder=-1, linewidth=0.8)
    
    rect_arr = []
    for itv in transcript.utr5s:
        start = itv.start
        rect_arr.append((start, len(itv), 0.1, 0.1))
    for itv in transcript.cdss:
        start = itv.start
        rect_arr.append((start, len(itv), 0.3, 0))
    for itv in transcript.utr3s:
        start = itv.start
        rect_arr.append((start, len(itv), 0.1, 0.1))
    
    end_arr = []
    for x in rect_arr:
        rect = Rectangle((x[0], h_start + x[3]), x[1], x[2], edgecolor='black', facecolor='black')
        ax.add_patch(rect)
        end_arr.append(x[0])
        end_arr.append(x[0] + x[1])

    if len(end_arr) == 0:
        return None

    x_arrows = np.linspace(max(min(end_arr), interval.start), min(max(end_arr), interval.end), num_arrows + 2)[1:-1]
    y_arrows = np.ones(num_arrows) * (h_start + 0.15)
    marker = ">" if interval.strand == "+" else "<"
    
    ax.scatter(x_arrows, y_arrows, c="lightgrey", marker=marker, s=(15 * scale_marker), zorder=10)
    if variant_location is not None:
        #if not isinstance(variant_location, list):
        #    variant_location = [variant_location]
        ax.scatter(variant_location, [h_start + 0.15] * len(variant_location), marker = "x", color = "orange", s=(15*scale_marker), zorder = 11)

def interval_from_string(interval, genome):

    interval = interval.replace(",", "")
    interval = interval.split(":")
    chrom = interval[0]
    strand = interval[2]
    ind = interval[1].split("-")

    interval = Interval(chrom, strand, int(ind[0]) - 1, int(ind[1]), genome)
    return interval

#TODO: Mark Variant location
def make_track_plot(interval, arr, gene, genome, hide_transcript=False, show_primary=True, figsize=(8, 2), height_scaling=4,
                    num_arrows=7, scale_marker=1.0, subplot_spacing=0.7,
                    transcript_spacing=0.5, colors=None, xlabel="", ylabels=None, ylims=None, hide_yticks=False,
                    absolute_scale=False, labelpad=20, variant_location=None, custom_transcript=None, vertical_bars=None, vertical_bars_colors=None,
                    vertical_markers=None, vertical_markers_colors=None,
                    superimpose=False, set_ymax=None, ylabel_loc="center",
                    alpha=1.0):
    if isinstance(interval, str):
        interval = interval_from_string(interval, genome)

    assert len(arr.shape) == 2
    assert len(interval) == arr.shape[0]

    if colors is not None:
        if isinstance(colors, str):
            colors = [colors] * arr.shape[1]
        else:
            assert len(colors) == arr.shape[1]
    else:
        colors = ["#1f77b4"] * arr.shape[1]

    if ylabels is not None:
        assert isinstance(ylabels, list)
        assert len(ylabels) == arr.shape[1]
    if ylims is not None:
        assert isinstance(ylims, list)
        assert len(ylims) == arr.shape[1]
        assert all([len(x) == 2 for x in ylims])
    if vertical_bars and vertical_bars_colors is not None:
        assert isinstance(vertical_bars, list)
        assert isinstance(vertical_bars_colors, list)
        assert len(vertical_bars) == len(vertical_bars_colors)
    
    height_ratios = [height_scaling] * arr.shape[1] + [1] if not hide_transcript else [height_scaling] * arr.shape[1]
    if superimpose:
        height_ratios = [height_scaling] * 1 + [1] if not hide_transcript else [height_scaling]

    nrows = arr.shape[1] + 1 if not hide_transcript else arr.shape[1]
    if superimpose:
        nrows = 1 + 1 if not hide_transcript else 1
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, sharex=True,
                           gridspec_kw={'height_ratios': height_ratios})
    xlim_left = interval.start
    xlim_right = interval.end
    x_vals = np.arange(xlim_left, xlim_right)

    if interval.strand == "-":
        arr = arr[::-1, :]

    for i in range(arr.shape[1]):
        if superimpose:
            ax_i = 0
        else:
            ax_i = i

        setup_track_axes(ax[ax_i])
        if ylims is not None:
            max_val = round(ylims[i][1], 3)
            min_val = round(ylims[i][0], 3)
        elif not absolute_scale:
            max_val = round(arr[:, i].max(), 3)
            min_val = round(arr[:, i].min(), 3)
        else:
            max_val = round(arr.max(), 3)
            min_val = round(arr.min(), 3)
        
        if set_ymax is not None:
            max_val = set_ymax
            
        ax[ax_i].set_ylim(top=max_val, bottom=min_val)
        if not hide_yticks:
            ax[ax_i].set_yticks([max_val])
        else:
            ax[ax_i].set_yticks([])
        ax[ax_i].set_xlim(left=xlim_left, right=xlim_right)
        ax[ax_i].plot(x_vals, arr[:, i], color=colors[i], alpha=alpha)
        ax[ax_i].fill_between(x_vals, arr[:, i], color=colors[i], alpha=alpha)
        if ylabels is not None:
            ax[ax_i].set_ylabel(ylabels[i], rotation=0, fontweight='semibold', fontsize=8, labelpad=labelpad, loc=ylabel_loc)
        if vertical_bars is not None:
            for vertical_bar, color in zip(vertical_bars, vertical_bars_colors):
                ax[ax_i].axvline(x=vertical_bar,ymin=-1.2,ymax=1,c=color,linewidth=0.5,zorder=0, clip_on=False)
    
    if not hide_transcript:
        if custom_transcript is not None:
            transcripts = [custom_transcript]
        elif show_primary:
            transcripts = sorted(gene.transcripts, key=lambda x: 100
                        if genome.appris_principality(x) is None
                        else genome.appris_principality(x),
                    )
            transcripts = [transcripts[0]]
        else:
            transcripts = gene.transcripts

        setup_genome_axes(ax[-1])
        ax[-1].set_xlim(left=xlim_left, right=xlim_right)
        h_start = 0
        h_start_arr = []
        for tx in transcripts:
            h_start_arr.append(h_start)
            add_transcript_rectangles(ax[-1], interval, tx, h_start, num_arrows=num_arrows,
                                    scale_marker=scale_marker, variant_location=variant_location)
            h_start += transcript_spacing

    if vertical_markers is not None:
        for vertical_marker, color in zip(vertical_markers, vertical_markers_colors):
            ax[-1].scatter(vertical_marker, [0.3], color=color, marker="v", s=40, zorder=10)
    
    plt.xlabel(xlabel, fontweight="semibold", fontsize=8)
    plt.subplots_adjust(hspace=subplot_spacing)

    return fig, ax