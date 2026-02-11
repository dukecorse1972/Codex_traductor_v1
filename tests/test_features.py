from __future__ import annotations

import numpy as np

from src.data.features import frame_to_feature, D_FRAME


class LM:
    def __init__(self, x, y, z, visibility=1.0, presence=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


class Category:
    def __init__(self, name, score=0.99):
        self.category_name = name
        self.score = score


def make_frame():
    pose_lms = [LM(0.1 + i * 0.001, 0.2 + i * 0.001, 0.0) for i in range(33)]
    left = [LM(0.3, 0.3, 0.0, 1.0, 1.0) for _ in range(21)]
    right = [LM(0.5, 0.5, 0.0, 1.0, 1.0) for _ in range(21)]

    frame = {
        "pose": {"pose_landmarks": [pose_lms]},
        "hands": {
            "hand_landmarks": [left, right],
            "handedness": [[Category("Left")], [Category("Right")]],
        },
        "holistic_legacy": None,
    }
    return frame


def test_feature_dimension_is_291():
    feat = frame_to_feature(make_frame())
    assert feat.shape == (D_FRAME,)
    assert D_FRAME == 291


def test_feature_order_deterministic():
    frame = make_frame()
    feat1 = frame_to_feature(frame)
    feat2 = frame_to_feature(frame)
    np.testing.assert_allclose(feat1, feat2, atol=1e-8)

    # left-hand slot starts right after pose block
    pose_dim = 33 * 5
    left_first_xyz = feat1[pose_dim: pose_dim + 3]
    right_first_xyz = feat1[pose_dim + 63: pose_dim + 66]
    assert not np.allclose(left_first_xyz, right_first_xyz)
