from __future__ import annotations

import io

from src.data.compat_pickle import MediaPipeCompatUnpickler


def test_find_class_creates_dummy_for_missing_mediapipe_framework_path():
    up = MediaPipeCompatUnpickler(io.BytesIO(b"N."))
    cls = up.find_class("mediapipe.framework.formats.landmark_pb2", "NormalizedLandmark")
    assert cls.__name__ == "NormalizedLandmark"
    obj = cls()
    # Dummy instances are flexible enough for getattr-based processing.
    assert obj is not None
