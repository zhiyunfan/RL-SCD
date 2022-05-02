#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2019 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr
# Camille Guinaudeau - https://sites.google.com/site/cguinaudeau/
# Mamadou Doumbia
# Diego Fustes diego.fustes at toptal.com

import numpy as np
from .core import Segment, Timeline, Annotation
from .core.utils.generators import pairwise

from .base import BaseMetric, f_measure
from .utils import UEMSupportMixin
from .core.segment import SlidingWindow
from .core.feature import SlidingWindowFeature
import scipy.signal

PURITY_NAME = 'segmentation purity'
COVERAGE_NAME = 'segmentation coverage'
PURITY_COVERAGE_NAME = 'segmentation F[purity|coverage]'
PTY_CVG_TOTAL = 'total duration'
PTY_CVG_INTER = 'intersection duration'

PTY_TOTAL = 'pty total duration'
PTY_INTER = 'pty intersection duration'
CVG_TOTAL = 'cvg total duration'
CVG_INTER = 'cvg intersection duration'

PRECISION_NAME = 'segmentation precision'
RECALL_NAME = 'segmentation recall'

PR_BOUNDARIES = 'number of boundaries'
PR_MATCHES = 'number of matches'


class SegmentationCoverage(BaseMetric):
    """Segmentation coverage

    Parameters
    ----------
    tolerance : float, optional
        When provided, preprocess reference by filling intra-label gaps shorter
        than `tolerance` (in seconds).

    """
    
    def __init__(self, tolerance=0.500, **kwargs):
        super(SegmentationCoverage, self).__init__(**kwargs)
        self.tolerance = tolerance

    def _partition(self, timeline, coverage):

        # boundaries (as set of timestamps)
        boundaries = set([])
        for segment in timeline:
            boundaries.add(segment.start)
            boundaries.add(segment.end)

        # partition (as timeline)
        partition = Annotation()
        
        for start, end in pairwise(sorted(boundaries)):
            segment = Segment(start, end)
            partition[segment] = '_'
        return partition.crop(coverage, mode='intersection').relabel_tracks()

    def _preprocess(self, reference, hypothesis, oracle_vad):

        if not isinstance(reference, Annotation):
            raise TypeError('reference must be an instance of `Annotation`')

        if isinstance(hypothesis, Annotation):
            hypothesis = hypothesis.get_timeline()

        # reference where short intra-label gaps are removed
        # 这个for循环，先对每一个说话人的所有segment去进行合并操作，合并的是那些，被短于tolerance长度的静音分割的segment，得到filled
        # 然后是将所有speaker放在一起进行取并集的操作得到coverage
        filled = Timeline()
        for label in reference.labels():
            label_timeline = reference.label_timeline(label)
            for gap in label_timeline.gaps():
                if gap.duration < self.tolerance:
                    label_timeline.add(gap)
                    

            for segment in label_timeline.support():
                filled.add(segment)

        # reference coverage after filling gaps
        
        coverage = filled.support()
        reference_partition = self._partition(filled, coverage)

        if oracle_vad == 'True':
            hypothesis_partition = self._partition(hypothesis, coverage)
        else:
            hypothesis_partition = self._partition(hypothesis, hypothesis)
        return reference_partition, hypothesis_partition

    def _process(self, reference, hypothesis):

        detail = self.init_components()

        # cooccurrence matrix
        K = reference * hypothesis
        detail[PTY_CVG_TOTAL] = np.sum(K).item()
        detail[PTY_CVG_INTER] = np.sum(np.max(K, axis=1)).item()

        return detail

    @classmethod
    def metric_name(cls):
        return COVERAGE_NAME

    @classmethod
    def metric_components(cls):
        return [PTY_CVG_TOTAL, PTY_CVG_INTER]

    def compute_components(self, reference, hypothesis, **kwargs):
        reference, hypothesis = self._preprocess(reference, hypothesis)
        return self._process(reference, hypothesis)

    def compute_metric(self, detail):
        return detail[PTY_CVG_INTER] / detail[PTY_CVG_TOTAL]


class SegmentationPurity(SegmentationCoverage):
    """Segmentation purity

    Parameters
    ----------
    tolerance : float, optional
        When provided, preprocess reference by filling intra-label gaps shorter
        than `tolerance` (in seconds).

    """

    @classmethod
    def metric_name(cls):
        return PURITY_NAME

    def compute_components(self, reference, hypothesis, **kwargs):
        reference, hypothesis = self._preprocess(reference, hypothesis)
        return self._process(hypothesis, reference)


class SegmentationPurityCoverageFMeasure(SegmentationCoverage):
    """
    Compute segmentation purity and coverage, and return their F-score.


    Parameters
    ----------
    tolerance : float, optional
        When provided, preprocess reference by filling intra-label gaps shorter
        than `tolerance` (in seconds).

    beta : float, optional
            When beta > 1, greater importance is given to coverage.
            When beta < 1, greater importance is given to purity.
            Defaults to 1.

    See also
    --------
    pyannote.metrics.segmentation.SegmentationPurity
    pyannote.metrics.segmentation.SegmentationCoverage
    pyannote.metrics.base.f_measure
    """

    def __init__(self, tolerance=0.500, beta=1, **kwargs):
        super(SegmentationPurityCoverageFMeasure, self).__init__(tolerance=tolerance, **kwargs)
        self.beta = beta

    def _process(self, reference, hypothesis, oracle_vad):
        reference, hypothesis = self._preprocess(reference, hypothesis, oracle_vad)

        detail = self.init_components()
        # cooccurrence matrix coverage
        K = reference * hypothesis
        detail[CVG_TOTAL] = np.sum(K).item()
        detail[CVG_INTER] = np.sum(np.max(K, axis=1)).item()

        # cooccurrence matrix purity
        detail[PTY_TOTAL] = detail[CVG_TOTAL]
        detail[PTY_INTER] = np.sum(np.max(K, axis=0)).item()
        return detail

    def compute_components(self, reference, hypothesis, oracle_vad, **kwargs):
        return self._process(reference, hypothesis, oracle_vad)

    def compute_metric(self, detail):
        _, _, value = self.compute_metrics(detail=detail)
        return value

    def compute_metrics(self, detail=None):
        detail = self.accumulated_ if detail is None else detail

        purity = \
            1. if detail[PTY_TOTAL] == 0. \
            else detail[PTY_INTER] / detail[PTY_TOTAL]

        coverage = \
            1. if detail[CVG_TOTAL] == 0. \
            else detail[CVG_INTER] / detail[CVG_TOTAL]
        #print('purity:', purity, 'coverage:', coverage, 'f_score:', f_measure(purity, coverage, beta=self.beta))

        return purity, coverage, f_measure(purity, coverage, beta=self.beta)

    @classmethod
    def metric_name(cls):
        return PURITY_COVERAGE_NAME

    @classmethod
    def metric_components(cls):
        return [PTY_TOTAL, PTY_INTER, CVG_TOTAL, CVG_INTER]


    def score2metric(self, score, laebel):
        score = np.random.rand(400)
        label = np.random.randint(0, 1, 400)
        sliding_window = SlidingWindow(duration=0.025, step=0.010, start=0.000)
        score = SlidingWindowFeature(score, sliding_window)
        label = SlidingWindowFeature(label, sliding_window)
        peak = Peak(0.5, 0.01)
        predict_segments = peak.apply(score)
        label_segments = peak.apply(label)
        metric = SegmentationPurityCoverageFMeasure()
        details = metric.compute_components(label_segments.to_annotation(), predict_segments, True)
        p, c, f = metric.compute_metrics(details)
        return p, c, f


class SegmentationPrecision(UEMSupportMixin, BaseMetric):
    """Segmentation precision

    >>> from pyannote.core import Timeline, Segment
    >>> from pyannote.metrics.segmentation import SegmentationPrecision
    >>> precision = SegmentationPrecision()

    >>> reference = Timeline()
    >>> reference.add(Segment(0, 1))
    >>> reference.add(Segment(1, 2))
    >>> reference.add(Segment(2, 4))

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 1))
    >>> hypothesis.add(Segment(1, 2))
    >>> hypothesis.add(Segment(2, 3))
    >>> hypothesis.add(Segment(3, 4))
    >>> precision(reference, hypothesis)
    0.6666666666666666

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 4))
    >>> precision(reference, hypothesis)
    1.0

    """

    @classmethod
    def metric_name(cls):
        return PRECISION_NAME

    @classmethod
    def metric_components(cls):
        return [PR_MATCHES, PR_BOUNDARIES]

    def __init__(self, tolerance=0., **kwargs):

        super(SegmentationPrecision, self).__init__(**kwargs)
        self.tolerance = tolerance

    def compute_components(self, reference, hypothesis, **kwargs):

        # extract timeline if needed
        if isinstance(reference, Annotation):
            reference = reference.get_timeline()
        if isinstance(hypothesis, Annotation):
            hypothesis = hypothesis.get_timeline()

        detail = self.init_components()

        # number of matches so far...
        nMatches = 0.  # make sure it is a float (for later ratio)

        # number of boundaries in reference and hypothesis
        N = len(reference) - 1
        M = len(hypothesis) - 1

        # number of boundaries in hypothesis
        detail[PR_BOUNDARIES] = M

        # corner case (no boundary in hypothesis or in reference)
        if M == 0 or N == 0:
            detail[PR_MATCHES] = 0.
            return detail

        # reference and hypothesis boundaries
        refBoundaries = [segment.end for segment in reference][:-1]
        hypBoundaries = [segment.end for segment in hypothesis][:-1]

        # temporal delta between all pairs of boundaries
        delta = np.zeros((N, M))
        for r, refBoundary in enumerate(refBoundaries):
            for h, hypBoundary in enumerate(hypBoundaries):
                delta[r, h] = abs(refBoundary - hypBoundary)

        # make sure boundaries too far apart from each other cannot be matched
        # (this is what np.inf is used for)
        delta[np.where(delta > self.tolerance)] = np.inf

        # h always contains the minimum value in delta matrix
        # h == np.inf means that no boundary can be matched
        h = np.amin(delta)

        # while there are still boundaries to match
        while h < np.inf:
            # increment match count
            nMatches += 1

            # find boundaries to match
            k = np.argmin(delta)
            i = k // M
            j = k % M

            # make sure they cannot be matched again
            delta[i, :] = np.inf
            delta[:, j] = np.inf

            # update minimum value in delta
            h = np.amin(delta)

        detail[PR_MATCHES] = nMatches
        return detail

    def compute_metric(self, detail):

        numerator = detail[PR_MATCHES]
        denominator = detail[PR_BOUNDARIES]

        if denominator == 0.:
            if numerator == 0:
                return 1.
            else:
                raise ValueError('')
        else:
            return numerator / denominator


class SegmentationRecall(SegmentationPrecision):
    """Segmentation recall

    >>> from pyannote.core import Timeline, Segment
    >>> from pyannote.metrics.segmentation import SegmentationRecall
    >>> recall = SegmentationRecall()

    >>> reference = Timeline()
    >>> reference.add(Segment(0, 1))
    >>> reference.add(Segment(1, 2))
    >>> reference.add(Segment(2, 4))

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 1))
    >>> hypothesis.add(Segment(1, 2))
    >>> hypothesis.add(Segment(2, 3))
    >>> hypothesis.add(Segment(3, 4))
    >>> recall(reference, hypothesis)
    1.0

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 4))
    >>> recall(reference, hypothesis)
    0.0

    """

    @classmethod
    def metric_name(cls):
        return RECALL_NAME

    def compute_components(self, reference, hypothesis, **kwargs):
        return super(SegmentationRecall, self).compute_components(
            hypothesis, reference)


class Peak(object):
    """Peak detection

    Parameters
    ----------
    alpha : float, optional
        Adaptative threshold coefficient. Defaults to 0.5
    scale : {'absolute', 'relative', 'percentile'}
        Set to 'relative' to make onset/offset relative to min/max.
        Set to 'percentile' to make them relative 1% and 99% percentiles.
        Defaults to 'absolute'.
    min_duration : float, optional
        Defaults to 1 second.
    log_scale : bool, optional
        Set to True to indicate that binarized scores are log scaled.
        Defaults to False.

    """

    def __init__(self, alpha=0.5, min_duration=1.0, scale="absolute", log_scale=False):
        super(Peak, self).__init__()
        self.alpha = alpha
        self.scale = scale
        self.min_duration = min_duration
        self.log_scale = log_scale

    def apply(self, predictions, dimension=0):
        """Peak detection

        Parameter
        ---------
        predictions : SlidingWindowFeature
            Predictions returned by segmentation approaches.

        Returns
        -------
        segmentation : Timeline
            Partition.
        """

        if len(predictions.data.shape) == 1:
            y = predictions.data
        elif predictions.data.shape[1] == 1:
            y = predictions.data[:, 0]
        else:
            y = predictions.data[:, dimension]

        if self.log_scale:
            y = np.exp(y)

        sw = predictions.sliding_window

        precision = sw.step
        order = max(1, int(np.rint(self.min_duration / precision)))
        indices = scipy.signal.argrelmax(y, order=order)[0]

        if self.scale == "absolute":
            mini = 0
            maxi = 1

        elif self.scale == "relative":
            mini = np.nanmin(y)
            maxi = np.nanmax(y)

        elif self.scale == "percentile":
            mini = np.nanpercentile(y, 1)
            maxi = np.nanpercentile(y, 99)

        threshold = mini + self.alpha * (maxi - mini)

        #print('threshold:', threshold, end=' ')

        peak_time = np.array([sw[i].middle for i in indices if y[i] > threshold])

        n_windows = len(y)
        start_time = sw[0].start
        end_time = sw[n_windows].end

        boundaries = np.hstack([[start_time], peak_time, [end_time]])
        segmentation = Timeline()
        for i, (start, end) in enumerate(pairwise(boundaries)):
            segment = Segment(start, end)
            segmentation.add(segment)

        return segmentation


if __name__ == '__main__':
    score = np.random.rand(400)
    label = np.random.randint(0, 1, 400)
    sliding_window = SlidingWindow(duration=0.025, step=0.010, start=0.000)
    score = SlidingWindowFeature(score, sliding_window)
    label = SlidingWindowFeature(label, sliding_window)
    peak = Peak(0.5, 0.01)
    predict_segments = peak.apply(score)
    label_segments = peak.apply(label)
    metric = SegmentationPurityCoverageFMeasure()
    details = metric.compute_components(label_segments.to_annotation(), predict_segments, True)
    p, c, f = metric.compute_metrics(details)
    print('purity, coverage, f_measure:', p, c, f)
