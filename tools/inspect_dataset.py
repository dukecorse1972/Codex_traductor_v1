#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from statistics import mean
import sys
import importlib
from types import ModuleType

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.features import FeatureSpec, D_FRAME


# -----------------------------------------------------------------------------
# Pickle compatibility for MediaPipe objects
# -----------------------------------------------------------------------------
# Your PKL files were created in an environment where classes lived under
# "mediapipe.framework.*". On Windows pip builds, that package path often
# doesn't exist (import mediapipe.framework fails).
#
# We handle this by remapping old module paths to new ones that exist in the
# current mediapipe package, and by providing dummies as a last resort so
# pickle can proceed.
# -----------------------------------------------------------------------------

def _get_or_create_module(name: str) -> ModuleType:
    """Import a module if possible; otherwise create an empty placeholder."""
    if name in sys.modules:
        return sys.modules[name]
    try:
        mod = importlib.import_module(name)
        return mod
    except Exception:
        mod = ModuleType(name)
        sys.modules[name] = mod
        return mod


class MediaPipeCompatUnpickler(pickle.Unpickler):
    """
    Unpickler that remaps old mediapipe module paths to current ones.

    Common old paths in pickles:
      - mediapipe.framework.formats.landmark_pb2
      - mediapipe.framework.formats.classification_pb2
      - mediapipe.framework.formats.detection_pb2
      - mediapipe.framework.formats.rect_pb2
    Current pip package typically exposes:
      - mediapipe.framework.formats.*  (sometimes missing on Windows)
      - mediapipe.tasks.python.components.containers.* (for tasks API)
      - or protobuf modules under mediapipe.framework.formats.* on Linux/Mac

    Strategy:
      1) Try to import the exact module name.
      2) If it starts with mediapipe.framework..., try replacing the prefix with
         mediapipe.framework.formats... etc (best effort).
      3) If still not possible, create a dummy module so unpickling can continue.
         This works for your inspection script because you only do getattr checks.
    """

    _PREFIX = "mediapipe.framework."

    def find_class(self, module: str, name: str):
        # First try normal behavior
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            pass
        except Exception:
            # Other pickle resolution issues: continue to remap attempts below
            pass

        # If it's an old mediapipe.framework.* path, try to remap
        if module.startswith(self._PREFIX):
            # Try importing the module as-is (some envs have it)
            mod = _get_or_create_module(module)

            # If the symbol exists in this module (real or dummy), return it
            if hasattr(mod, name):
                return getattr(mod, name)

            # Best-effort remaps for common protobuf modules
            remap_candidates = [
                # Often the same path works on other platforms
                module,

                # Sometimes pickles store "mediapipe.framework.formats.*" but
                # current install needs the same; we just retry via importer.
                module.replace("mediapipe.framework.", "mediapipe.framework."),

                # Rare: old path might include extra nesting; keep as is.
            ]

            for m in remap_candidates:
                try:
                    real_mod = importlib.import_module(m)
                    if hasattr(real_mod, name):
                        return getattr(real_mod, name)
                except Exception:
                    continue

            # Last resort: create dummy class so pickle can instantiate it
            dummy_mod = _get_or_create_module(module)
            Dummy = type(name, (), {})
            setattr(dummy_mod, name, Dummy)
            return Dummy

        # Not mediapipe-related: re-raise the original error contextually
        raise ModuleNotFoundError(
            f"Cannot import '{module}'. Missing module while unpickling '{name}'. "
            f"If this comes from a different dependency, install it or regenerate the PKL."
        )


def load_pickle_compat(pkl_path: Path):
    """Load pickle with MediaPipe-compatible unpickling."""
    with open(pkl_path, "rb") as f:
        return MediaPipeCompatUnpickler(f).load()


# -----------------------------------------------------------------------------
# Inspection logic
# -----------------------------------------------------------------------------

def _safe_getattr(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def inspect_sample(pkl_path: Path):
    frames = load_pickle_compat(pkl_path)

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
