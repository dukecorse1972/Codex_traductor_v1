from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import math
import random

import numpy as np

# MediaPipe pose indices (BlazePose)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
POSE_DIM_PER_LM = 5  # x, y, z, visibility, presence
HAND_DIM_PER_LM = 3  # x, y, z
D_FRAME = POSE_LANDMARKS * POSE_DIM_PER_LM + 2 * HAND_LANDMARKS * HAND_DIM_PER_LM


@dataclass
class FeatureSpec:
    d_frame: int = D_FRAME
    t_fixed: int = 128
    pose_order: str = "pose_landmarks[0] with 33 landmarks, each [x,y,z,visibility,presence]"
    hands_order: str = "fixed Left then Right slots, each 21 landmarks [x,y,z], zero-filled if missing"
    normalization: str = (
        "Body-centered: subtract body center (mid hips if both visible/present, else mid shoulders) "
        "from pose xyz and hand xyz, then scale xyz by shoulder distance if available (>1e-4), otherwise 1.0."
    )
    padding_policy: str = "pad_with_zeros_at_end"
    truncation_policy: str = "uniform_subsample_if_longer_than_t_fixed"
    temporal_augmentations: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.temporal_augmentations is None:
            self.temporal_augmentations = {
                "temporal_dropout_prob": 0.1,
                "temporal_dropout_ratio": 0.1,
                "gaussian_jitter_std": 0.002,
                "landmark_dropout_prob": 0.05,
                "landmark_dropout_ratio": 0.03,
            }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(path: str) -> "FeatureSpec":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FeatureSpec(**data)


def _safe_getattr(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _landmark_to_array(lm: Any, include_vis_presence: bool = False) -> np.ndarray:
    x = float(_safe_getattr(lm, "x", 0.0))
    y = float(_safe_getattr(lm, "y", 0.0))
    z = float(_safe_getattr(lm, "z", 0.0))
    if include_vis_presence:
        v = float(_safe_getattr(lm, "visibility", 0.0))
        p = float(_safe_getattr(lm, "presence", 0.0))
        return np.array([x, y, z, v, p], dtype=np.float32)
    return np.array([x, y, z], dtype=np.float32)


def _extract_pose_lms(frame: Dict[str, Any]) -> List[Any]:
    pose = frame.get("pose")
    pose_landmarks = _safe_getattr(pose, "pose_landmarks", [])
    if pose_landmarks and len(pose_landmarks) > 0:
        seq = pose_landmarks[0]
        if isinstance(seq, Sequence):
            return list(seq)
    return []


def _extract_hand_slots(frame: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
    """Return fixed (left_lms, right_lms), each len<=21 landmark objects."""
    hands = frame.get("hands")
    hand_landmarks = _safe_getattr(hands, "hand_landmarks", []) or []
    handedness = _safe_getattr(hands, "handedness", []) or []

    left, right = [], []
    used = set()

    for i, hd in enumerate(handedness):
        cat = None
        if isinstance(hd, Sequence) and len(hd) > 0:
            cat = hd[0]
        else:
            cat = hd
        cname = str(_safe_getattr(cat, "category_name", "")).lower()
        if i < len(hand_landmarks):
            if cname == "left" and not left:
                left = list(hand_landmarks[i])
                used.add(i)
            elif cname == "right" and not right:
                right = list(hand_landmarks[i])
                used.add(i)

    # Fallback: assign remaining by order if ambiguous/missing
    remaining_idx = [i for i in range(len(hand_landmarks)) if i not in used]
    for i in remaining_idx:
        if not left:
            left = list(hand_landmarks[i])
        elif not right:
            right = list(hand_landmarks[i])
        if left and right:
            break

    return left[:HAND_LANDMARKS], right[:HAND_LANDMARKS]


def _body_center_and_scale(pose_vec: np.ndarray) -> Tuple[np.ndarray, float]:
    """pose_vec shape [33,5]. Return center xyz and shoulder scale."""
    if pose_vec.shape[0] < POSE_LANDMARKS:
        return np.zeros(3, dtype=np.float32), 1.0

    lh = pose_vec[LEFT_HIP]
    rh = pose_vec[RIGHT_HIP]
    ls = pose_vec[LEFT_SHOULDER]
    rs = pose_vec[RIGHT_SHOULDER]

    hips_ok = lh[3] > 0.3 and rh[3] > 0.3 and lh[4] > 0.3 and rh[4] > 0.3
    shoulders_ok = ls[3] > 0.3 and rs[3] > 0.3 and ls[4] > 0.3 and rs[4] > 0.3

    if hips_ok:
        center = (lh[:3] + rh[:3]) / 2.0
    elif shoulders_ok:
        center = (ls[:3] + rs[:3]) / 2.0
    else:
        center = np.zeros(3, dtype=np.float32)

    shoulder_dist = float(np.linalg.norm(ls[:3] - rs[:3]))
    scale = shoulder_dist if shoulder_dist > 1e-4 else 1.0
    return center.astype(np.float32), scale


def frame_to_feature(frame: Dict[str, Any]) -> np.ndarray:
    pose_lms = _extract_pose_lms(frame)

    pose_arr = np.zeros((POSE_LANDMARKS, POSE_DIM_PER_LM), dtype=np.float32)
    for i in range(min(len(pose_lms), POSE_LANDMARKS)):
        pose_arr[i] = _landmark_to_array(pose_lms[i], include_vis_presence=True)

    center, scale = _body_center_and_scale(pose_arr)
    pose_xyz = (pose_arr[:, :3] - center[None, :]) / scale
    pose_arr[:, :3] = pose_xyz

    left_lms, right_lms = _extract_hand_slots(frame)
    left_arr = np.zeros((HAND_LANDMARKS, HAND_DIM_PER_LM), dtype=np.float32)
    right_arr = np.zeros((HAND_LANDMARKS, HAND_DIM_PER_LM), dtype=np.float32)

    for i in range(min(len(left_lms), HAND_LANDMARKS)):
        left_arr[i] = _landmark_to_array(left_lms[i], include_vis_presence=False)
    for i in range(min(len(right_lms), HAND_LANDMARKS)):
        right_arr[i] = _landmark_to_array(right_lms[i], include_vis_presence=False)

    left_arr = (left_arr - center[None, :]) / scale
    right_arr = (right_arr - center[None, :]) / scale

    feat = np.concatenate([
        pose_arr.reshape(-1),
        left_arr.reshape(-1),
        right_arr.reshape(-1),
    ], axis=0).astype(np.float32)

    if feat.shape[0] != D_FRAME:
        raise ValueError(f"Unexpected feature dim {feat.shape[0]} != {D_FRAME}")
    return feat


def _uniform_subsample_indices(length: int, t_fixed: int) -> np.ndarray:
    return np.linspace(0, length - 1, t_fixed).round().astype(int)


def sequence_to_fixed(
    frames: Sequence[Dict[str, Any]],
    t_fixed: int,
    augment: bool = False,
    rng: Optional[random.Random] = None,
    temporal_dropout_prob: float = 0.1,
    temporal_dropout_ratio: float = 0.1,
    gaussian_jitter_std: float = 0.002,
    landmark_dropout_prob: float = 0.05,
    landmark_dropout_ratio: float = 0.03,
) -> np.ndarray:
    if rng is None:
        rng = random

    feats = [frame_to_feature(fr) for fr in frames]
    if len(feats) == 0:
        return np.zeros((t_fixed, D_FRAME), dtype=np.float32)

    seq = np.stack(feats, axis=0)  # [T,D]

    if augment:
        # Temporal dropout (drop random frames by masking to zero)
        if rng.random() < temporal_dropout_prob and seq.shape[0] > 2:
            n_drop = max(1, int(seq.shape[0] * temporal_dropout_ratio))
            drop_idx = rng.sample(range(seq.shape[0]), k=min(n_drop, seq.shape[0] - 1))
            seq[drop_idx] = 0.0

        # Add Gaussian jitter only on coordinate channels (all dims except vis/presence dims in pose)
        if gaussian_jitter_std > 0:
            noise = np.random.normal(0, gaussian_jitter_std, size=seq.shape).astype(np.float32)
            pose_flat = POSE_LANDMARKS * POSE_DIM_PER_LM
            mask = np.ones((D_FRAME,), dtype=np.float32)
            for i in range(POSE_LANDMARKS):
                vis_idx = i * POSE_DIM_PER_LM + 3
                pres_idx = i * POSE_DIM_PER_LM + 4
                mask[vis_idx] = 0.0
                mask[pres_idx] = 0.0
            seq += noise * mask[None, :]

        # Landmark dropout: zero random landmark chunks
        if rng.random() < landmark_dropout_prob:
            n_landmarks = POSE_LANDMARKS + 2 * HAND_LANDMARKS
            n_drop = max(1, int(n_landmarks * landmark_dropout_ratio))
            for idx in rng.sample(range(n_landmarks), k=min(n_drop, n_landmarks)):
                if idx < POSE_LANDMARKS:
                    s = idx * POSE_DIM_PER_LM
                    e = s + POSE_DIM_PER_LM
                else:
                    hidx = idx - POSE_LANDMARKS
                    s = POSE_LANDMARKS * POSE_DIM_PER_LM + hidx * HAND_DIM_PER_LM
                    e = s + HAND_DIM_PER_LM
                seq[:, s:e] = 0.0

    t = seq.shape[0]
    if t > t_fixed:
        idx = _uniform_subsample_indices(t, t_fixed)
        seq = seq[idx]
    elif t < t_fixed:
        pad = np.zeros((t_fixed - t, D_FRAME), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)

    return seq.astype(np.float32)
