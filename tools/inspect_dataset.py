#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import random
from statistics import mean

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.features import FeatureSpec, D_FRAME


def _safe_getattr(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def inspect_sample(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)

    if not isinstance(frames, list):
        raise ValueError(f"{pkl_path} expected list, got {type(frames)}")

    t = len(frames)
    hand_missing = 0
    pose_consistency = True

    for fr in frames:
        hands = fr.get("hands") if isinstance(fr, dict) else None
        hand_landmarks = _safe_getattr(hands, "hand_landmarks", []) or []
        if len(hand_landmarks) == 0:
            hand_missing += 1

        pose = fr.get("pose") if isinstance(fr, dict) else None
        pose_landmarks = _safe_getattr(pose, "pose_landmarks", []) or []
        if not pose_landmarks:
            pose_consistency = False
            continue
        seq = pose_landmarks[0]
        if len(seq) != 33:
            pose_consistency = False
            continue
        first = seq[0]
        attrs = [hasattr(first, a) for a in ["x", "y", "z", "visibility", "presence"]]
        if not all(attrs):
            pose_consistency = False

    missing_ratio = hand_missing / t if t > 0 else 0.0
    return t, missing_ratio, pose_consistency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--mediapipe_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--output", type=str, default="artifacts/feature_spec.json")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    mediapipe_dir = Path(args.mediapipe_dir)

    train_csv = splits_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}")

    df = pd.read_csv(train_csv, header=None, names=["FILENAME", "CLASS_ID"])
    df["FILENAME"] = df["FILENAME"].astype(str)

    n = min(args.n_samples, len(df))
    sample_rows = df.sample(n=n, random_state=42) if n > 0 else df

    Ts = []
    missing_hand_ratios = []
    pose_flags = []

    print(f"Inspecting {n} random samples from {train_csv} ...")
    for _, row in tqdm(sample_rows.iterrows(), total=n):
        pkl_path = mediapipe_dir / f"{row['FILENAME']}.pkl"
        if not pkl_path.exists():
            print(f"[WARN] missing pkl: {pkl_path}")
            continue
        try:
            t, miss, pose_ok = inspect_sample(pkl_path)
            Ts.append(t)
            missing_hand_ratios.append(miss)
            pose_flags.append(pose_ok)
        except Exception as exc:
            print(f"[WARN] could not inspect {pkl_path}: {exc}")

    if not Ts:
        raise RuntimeError("No valid samples inspected. Check paths and pkl files.")

    p95 = int(np.percentile(np.array(Ts), 95))
    t_fixed = min(max(p95, 16), 160)

    print("--- Inspection summary ---")
    print(f"Samples inspected: {len(Ts)}")
    print(f"T stats -> min: {min(Ts)} | max: {max(Ts)} | mean: {mean(Ts):.2f} | p95: {p95}")
    print(f"Frames without hands (% average): {100.0 * mean(missing_hand_ratios):.2f}%")
    print(f"Pose consistency (33 landmarks + attrs) all true?: {all(pose_flags)}")
    if not all(pose_flags):
        print("[INFO] Found pose inconsistencies; feature extractor will robustly zero-fill missing entries.")

    spec = FeatureSpec(t_fixed=t_fixed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec.to_json(str(out_path))

    print(f"Saved feature spec to {out_path}")
    print(json.dumps({
        "D_frame": D_FRAME,
        "T_fixed": t_fixed,
        "padding": spec.padding_policy,
        "truncation": spec.truncation_policy,
        "normalization": spec.normalization,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
