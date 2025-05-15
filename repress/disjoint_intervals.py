# Copyright (2025) Deep Genomics Incorporated All rights reserved - no unauthorized use or reproduction
# Licensed under CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Please otherwise contact legal@deepgenomics.com

import genome_kit as gk

# These dummy values are used to construct the intervals in the
# coordinate system of the DisjointIntervalsSequence.
DummyChr = "chr1"
DummyRefg = "hg19"


class DisjointIntervalsSequence(object):
    """A set of disjoint intervals against a base genome.

    Represents a DNA sequence corresponding to a set of disjoint
    intervals.  An object of this type can be sliced and the DNA
    sequence can be evaluated via the `dna()` method, just like a
    `genome_kit.Genome` object.  Objects of this class can be used to
    represent a strand of mRNA.  The coordinate system used with these
    objects uses an arbitrary chromosome (hack!) and numbers the
    nucleotide positions implied by the set of intervals contiguously
    such that the start of the lifted interval is 0.  The strand of
    the lifted coordinate system is the same as the strand of the
    disjoint intervals.  Use the `lift_interval()` method to lift an
    interval in the coordinate system of the base genome to the
    coordinate system used by this object.

    """

    def __init__(self, intervals, base_genome):
        """Initialize from a list of intervals and a base genome.

        The intervals must all be on the same chromosome and strand
        and they must be non-overlapping.  The list of intervals
        stored in the sequence object might be shorter than the input
        because intervals in the input that are adjacent are merged.

        Parameters
        ----------

        intervals : :py:class:`list` of :py:class:`genome_kit.Interval`
            A list of non-overlapping intervals on the same chromosome.

        base_genome : :py:class:`genome_kit.Genome` | :py:class:`genome_kit.VariantGenome`
            The genome object indexed by the intervals.

        """
        assert isinstance(intervals, list)
        assert len(intervals) > 0
        assert all(isinstance(x, gk.Interval) for x in intervals)

        self._check_consistent(intervals)

        # check that they have non-zero length
        assert all([len(ival) > 0 for ival in intervals])

        # sort 5' -> 3'
        intervals = sorted(intervals, key=lambda x: x.as_rna1())

        # check that they do not overlap
        assert not any([x.overlaps(y) for x, y in zip(intervals[:-1], intervals[1:])])

        # merge adjacent contiguous intervals
        intervals = self._merge_contiguous(intervals)

        # set the attributes
        self.base_genome = base_genome
        self.intervals = intervals

        length = sum([len(x) for x in self.intervals])
        self.interval = gk.Interval(DummyChr, self._strand(), 0, length, DummyRefg)

    @staticmethod
    def _check_consistent(intervals):
        # check that they have the same chr and strand
        ival0 = intervals[0]
        assert all([ival.chromosome == ival0.chromosome for ival in intervals])
        assert all([ival.strand == ival0.strand for ival in intervals])

        # check that they are consistently anchored
        assert all([ival.anchor == ival0.anchor for ival in intervals])
        assert all([ival.anchor_offset == ival0.anchor_offset for ival in intervals])

    @staticmethod
    def _merge_contiguous(intervals):
        merged_intervals = []
        last = intervals[0]
        for ival in intervals[1:]:
            if last.end3.end == ival.end5.end:
                last = gk.Interval.spanning(last, ival)
            else:
                merged_intervals.append(last)
                last = ival
        merged_intervals.append(last)
        return merged_intervals

    def __getitem__(self, item):
        """Slice with an interval or regular slice object.

        The sequence can be sliced with a `genome_kit.Interval` object or
        as usual in python (step == 1 only).  In either case, the
        coordinates are in the contiguous coordinate system induced by
        the intervals.

        """
        # you can index with a slice or with an interval
        if isinstance(item, slice):
            if item.start < 0 or item.stop > len(self.interval):
                raise IndexError("Invalid DisjointIntervalsSequence slice: {}".format(item))
            if item.start == item.stop:
                raise IndexError("DisjointIntervalsSequence can not be zero-sliced")

            start = item.start
            end = item.stop
        elif isinstance(item, gk.Interval):
            if item.strand != self.interval.strand:
                raise IndexError("Strand of interval {} slicing DisjointIntervalsSequence does not match".format(item))

            if not item.within(self.interval):
                raise IndexError("Invalid interval of slicing DisjointIntervalsSequence: {}".format(item))

            start = item.start
            end = item.end
        else:
            raise IndexError("Cannot slice DisjointIntervalsSequence with {} object".format(type(item)))

        ivals = []
        running_sum = 0
        for ival in self.intervals:
            y0, y1 = self._from_base_interval(ival)
            x0 = y0 + start - running_sum
            x1 = y0 + end - running_sum
            z0, z1 = self._intersect(x0, x1, y0, y1)
            if z0 != None:
                offset_start = z0 - y0
                offset_end = z1 - y0
                ival2 = self._to_base_interval(ival, offset_start, offset_end)
                ivals.append(ival2)
            running_sum += y1 - y0

        return DisjointIntervalsSequence(ivals, self.base_genome)

    def __len__(self):
        """Return the length of the sequence."""
        return len(self.interval)

    def __iter__(self):
        raise TypeError(f"'{type(self)}' object is not iterable")

    def __add__(self, other):
        """Sequences can be concatenated together with `+`

        The base genomes must be the same and the intervals on the
        second sequence must be downstream of the intervals of first
        sequence.
        """
        if not isinstance(other, DisjointIntervalsSequence):
            raise ValueError("attempting to concatenate {} with DisjointIntervalsSequence".format(str(type(other))))

        # validate the genomes
        if self.base_genome != other.base_genome:
            raise ValueError("DisjointIntervalsSequences must have same base genomes to concatenate")

        # Validate ordering of self and other.  While we could allow
        # concatenation where the intervals of self and other end up
        # being interleaved in the result, I don't think it is usually
        # meaningful.
        if not self.intervals[-1].upstream_of(other.intervals[0]):
            raise ValueError("attempting to concatenate other DisjointIntervalsSequence that is not downstream")

        # other checks are performed during initialization
        return DisjointIntervalsSequence(self.intervals + other.intervals,
                                         self.base_genome)

    def dna(self, ival):
        """Extract a DNA sequence.

        The specified interval is overlapped with the DNA strings of
        the disjoint intervals and the results are extracted from the
        base genome and concatenated.

        Parameters
        ----------

        ival : :py:class:`genome_kit.Interval`
            The interval for which to return the DNA string.

        Returns
        -------
        :py:class:`str`
            A DNA sequence.

        """
        assert ival.chromosome == DummyChr
        assert ival.reference_genome == DummyRefg

        # check for the opposite strand
        opposite = ival.strand != self._strand()
        if opposite:
            ival = ival.as_opposite_strand()

        assert ival.within(self.interval)
        dna = ""
        running_sum = 0
        x0, x1 = self._to_offsets(ival)
        for ival2 in self.intervals:
            y0 = running_sum
            y1 = y0 + len(ival2)
            if y0 > x1:
                break
            z0, z1 = self._intersect(x0, x1, y0, y1)
            if z0 != None:
                start_offset = z0 - y0
                end_offset = z1 - y0
                ival3 = self._to_base_interval(ival2, start_offset, end_offset)
                if opposite:
                    ival_dna = self.base_genome.dna(ival3.as_opposite_strand())
                    dna = ival_dna + dna
                else:
                    dna += self.base_genome.dna(ival3)
            running_sum = y1

        return dna

    def lift_interval(self, ival):
        """Transform an interval (or list of intervals) from the base genome.

        An interval (or a list of intervals) in the coordinate system
        of the base genome is transformed to the coordinate system of
        this object.  If a list of intervals is specified, the
        intervals must be consistently stranded, non-overlapping, and
        sorted 5' to 3'.  Also, they must transform to a single
        contiguous interval.

        Parameters
        ----------

        ival : :py:class:`genome_kit.Interval` | :py:class:`list` of :py:class:`genome_kit.Interval`
            The interval(s) in the base genome.

        Returns
        -------
        :py:class:`genome_kit.Interval`
            The interval in the coordinate system of this object.

        """
        # we handling opposite strand at this level
        if isinstance(ival, gk.Interval):
            opposite = ival.strand != self._strand()
            if opposite:
                ival = ival.as_opposite_strand()

            result = self._lift_interval(ival)

        elif isinstance(ival, list):
            # check that they are consistent
            ivals = ival
            self._check_consistent(ivals)

            opposite = ivals[0].strand != self._strand()
            if opposite:
                ivals = [ival.as_opposite_strand() for ival in ivals]
                ivals.reverse()

            # lift them all
            ivals = [self._lift_interval(ival) for ival in ivals]

            # merge
            ivals = self._merge_contiguous(ivals)

            # extract the result
            assert len(ivals) == 1
            result = ivals[0]

        else:
            raise TypeError("Invalid argument to lift_interval(): {}".format(ival))

        # handle strand
        if opposite:
            result = result.as_opposite_strand()
        return result

    def _lift_interval(self, ival):
        assert isinstance(ival, gk.Interval)
        assert ival.chromosome == self.intervals[0].chromosome
        assert ival.strand == self._strand()

        running_sum = 0
        for ival2 in self.intervals:
            if ival.within(ival2):
                x0, x1 = self._from_base_interval(ival)
                y0, y1 = self._from_base_interval(ival2)
                start = running_sum + x0 - y0
                end = running_sum + x1 - y0
                return self._from_offsets(start, end)

            running_sum += len(ival2)

        raise ValueError("Interval {} not contained in disjoint intervals".format(ival))

    def lower_interval(self, interval):
        """Transform an interval to the base genome.

        An interval in the coordinate system of this sequence is
        transformed into an ordered list of intervals in the
        coordinate system of the base genome.  The interval must be
        contained in the space of this sequence.  The result is a list
        because a single interval in the coordinates of the sequence
        can map to several of the disjoint base intervals.

        Suppose that a sequence is created from two intervals (only
        the coordinates are shown): (100,110) and (120,130).  This
        sequence has a total length of 20.  The zero length interval
        (10,10) in the coordinates of the sequence represents the site
        between the two base intervals.  This interval could therefore
        be mapped either to (110,110) or (120,120) by this method.
        This method will always map such ambiguous intervals to the
        upstream interval, so the result would be [(110,110)].

        Furthermore, a non-zero length input interval will never
        result in the output including a zero length interval in the
        output of this method.  So, given the same sequence as above,
        the input (10,15) will result in [(120,125)].

        Parameters
        ----------
        interval : :py:class:`genome_kit.Interval`
            The interval to be translated to the base genome.

        Returns
        -------
        list of :py:class:`genome_kit.Interval`
            The list of intervals that represents the projection of
            the input interval onto the coordinate system of the base
            genome.

        """
        # check the chromosome and genome
        if interval.chrom != DummyChr or interval.reference_genome != DummyRefg:
            raise ValueError("Invalid coordinate system for argument to lower_interval(): {}".format(interval))

        # check the strand
        opposite = interval.strand != self._strand()
        if opposite:
            interval = interval.as_opposite_strand()

        # check the size
        if not interval.within(self.interval):
            raise ValueError("Invalid extents for argument to lower_interval(): {}".format(interval))

        # lower the interval
        ivals = self._lower_interval(interval)

        # reverse if necessary
        if opposite:
            ivals = [ival.as_opposite_strand() for ival in ivals]
            ivals.reverse()

        return ivals

    def _lower_interval(self, interval):
        ivals = []
        running_sum = 0
        x0, x1 = self._to_offsets(interval)

        # handle 0-length intervals separately
        if x0 == x1:
            for ival2 in self.intervals:
                y0 = running_sum
                y1 = y0 + len(ival2)
                if x0 <= y1:
                    offset = x0 - y0
                    ival3 = self._to_base_interval(ival2, offset, offset)
                    return [ival3]

                running_sum = y1
            assert False # should not be reachable

        # handle non-zero length intervals
        for ival2 in self.intervals:
            y0 = running_sum
            y1 = y0 + len(ival2)
            if y0 > x1:
                break
            z0, z1 = self._intersect(x0, x1, y0, y1)
            if z0 != None:
                start_offset = z0 - y0
                end_offset = z1 - y0
                ival3 = self._to_base_interval(ival2, start_offset, end_offset)
                ivals.append(ival3)
            running_sum = y1

        return ivals

    def _strand(self):
        return self.intervals[0].strand

    # convert a lifted interval to offsets
    def _to_offsets(self, ival):
        if ival.strand == "+":
            return ival.start, ival.end
        else:
            length = len(self)
            x0 = length - ival.end
            x1 = length - ival.start
            return x0, x1

    # convert offsets to a lifted interval
    def _from_offsets(self, x0, x1):
        strand = self.interval.strand
        if strand == "+":
            start = x0
            end = x1
        else:
            length = len(self)
            start = length - x1
            end = length - x0
        return gk.Interval(DummyChr, strand, start, end, DummyRefg)

    @staticmethod
    def _from_base_interval(ival):
        if ival.strand == "+":
            return ival.start, ival.end
        else:
            return -ival.end, -ival.start

    @staticmethod
    def _to_base_interval(ival, offset_start, offset_end):
        if ival.strand == "+":
            start = ival.start + offset_start
            end = ival.start + offset_end
        else:
            start = ival.end - offset_end
            end = ival.end - offset_start
        return gk.Interval(ival.chromosome, ival.strand, start, end, ival.reference_genome)


    @staticmethod
    def _intersect(x0, x1, y0, y1):
        # check for empty intersection
        if x1 <= y0 or x0 >= y1:
            return None, None

        return max(x0, y0), min(x1, y1)

    @property
    def midpoint(self):
        """Identify the midpoint of a disjoint interval on the disjoint interval coordinate,
        return value in the original coordinate system

        Returns
        -------
        :py:class:`genome_kit.Interval`
            The midpoint len-0 (even) or len-1 (odd) interval, in the original coordinate system
        """
        # distinguish two cases:
        if len(self.interval) % 2 == 0:
            # case 1: even length, the actual midpoint is a len-0 interval,
            # but len-0 interval cannot be lower back to original coord (due to ambiguity at 'break points')
            # we extend it downstream to len-1, lower, then take the 5' end
            mid_itv_lifted = self.interval.end5.shift(len(self.interval) // 2).expand(0, 1)
            mid_itv = self.lower_interval(mid_itv_lifted)
            assert len(mid_itv) == 1
            return mid_itv[0].end5
        else:
            # case 2: odd length, the actual midpoint is a len-1 interval
            mid_itv_lifted = self.interval.end5.shift(len(self.interval) // 2).expand(0, 1)
            mid_itv = self.lower_interval(mid_itv_lifted)
            assert len(mid_itv) == 1
            return mid_itv[0]


def get_utr_cds_exon(transcript, genome):
    """Construct 3 DisjointIntervalsSequence objects for a protein coding transcript.

    Parameters
    ----------
    transcript : :py:class:`genome_kit.Transcript`
        The transcript of interest.
    genome : :py:class:`genome_kit.Genome`
        The genome of the transcript.

    Returns
    -------
    :py:class:`~modelzoo.utils.DisjointIntervalsSequence` | None
        A sequence of the UTR5 intervals.
    :py:class:`~modelzoo.utils.DisjointIntervalsSequence` | None
        A sequence of the CDS intervals.
    :py:class:`~modelzoo.utils.DisjointIntervalsSequence` | None
        A sequence of the UTR3 intervals.

    """
    assert transcript.cdss, "Only works for protein coding transcripts (transcript.cdss needs to be a non-empty list)."
    list_utr5 = []
    list_cds_exon = [x.interval for x in transcript.cdss]
    list_utr3 = []
    for exon in transcript.exons:
        if exon.cds is None:
            if exon.interval.upstream_of(transcript.cdss[0]):
                list_utr5.append(exon.interval)
            elif exon.interval.dnstream_of(transcript.cdss[-1]):
                list_utr3.append(exon.interval)
            else:
                raise ValueError
        elif exon.cds.interval != exon.interval:
            if exon.strand == '+' and exon.interval.start < exon.cds.interval.start:
                list_utr5.append(gk.Interval(chromosome=exon.interval.chromosome, strand=exon.interval.strand,
                                             start=exon.interval.start, end=exon.cds.interval.start,
                                             reference_genome=exon.interval.reference_genome))
            elif exon.strand == '-' and exon.interval.end > exon.cds.interval.end:
                list_utr5.append(gk.Interval(chromosome=exon.interval.chromosome, strand=exon.interval.strand,
                                             start=exon.cds.interval.end, end=exon.interval.end,
                                             reference_genome=exon.interval.reference_genome))

            if exon.strand == '+' and exon.interval.end > exon.cds.interval.end:
                list_utr3.append(gk.Interval(chromosome=exon.interval.chromosome, strand=exon.interval.strand,
                                             start=exon.cds.interval.end, end=exon.interval.end,
                                             reference_genome=exon.interval.reference_genome))
            elif exon.strand == '-' and exon.interval.start < exon.cds.interval.start:
                list_utr3.append(gk.Interval(chromosome=exon.interval.chromosome, strand=exon.interval.strand,
                                             start=exon.interval.start, end=exon.cds.interval.start,
                                             reference_genome=exon.interval.reference_genome))
            #else:
            #    raise ValueError
        else:
            pass

    # return 3 DisjointIntervalsSequence's
    seqs = []
    for ivals in [list_utr5, list_cds_exon, list_utr3]:
        if len(ivals) > 0:
            seqs.append(DisjointIntervalsSequence(ivals, genome))
        else:
            seqs.append(None)

    return tuple(seqs)


__all__ = ["DisjointIntervalsSequence", "get_utr_cds_exon"]