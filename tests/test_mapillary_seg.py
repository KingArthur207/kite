"""The Mapillary SegFormer checkpoint loads into the installed transformers SegFormer."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "third_party", "videotesting")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

WEIGHTS = os.path.join(VENDOR, "weights", "segmentation", "SegFormer_b0_Mapillary.pth")


@pytest.mark.skipif(not os.path.isfile(WEIGHTS), reason="Mapillary SegFormer weights are not on disk")
def test_mapillary_checkpoint_loads_and_labels_a_frame():
    from streetseg.mapillary_seg import MapillarySegFormer

    segmenter = MapillarySegFormer(WEIGHTS)
    labels = segmenter.predict(np.zeros((64, 64, 3), np.uint8))
    assert labels.shape == (64, 64)
    assert labels.dtype == np.int32
    assert int(labels.min()) >= 0
    assert int(labels.max()) < len(segmenter.names)
