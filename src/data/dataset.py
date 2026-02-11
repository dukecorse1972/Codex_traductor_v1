from __future__ import annotations

from pathlib import Path
import pickle
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import sequence_to_fixed, FeatureSpec


class SWLLSEDataset(Dataset):
    def __init__(
        self,
        split_csv: str,
        mediapipe_dir: str,
        feature_spec: FeatureSpec,
        augment: bool = False,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.mediapipe_dir = Path(mediapipe_dir)
        self.feature_spec = feature_spec
        self.augment = augment

        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv}")
        if not self.mediapipe_dir.exists():
            raise FileNotFoundError(f"Mediapipe dir not found: {self.mediapipe_dir}")

        self.df = pd.read_csv(self.split_csv, header=None, names=["FILENAME", "CLASS_ID"])
        self.df["FILENAME"] = self.df["FILENAME"].astype(str)
        self.df["CLASS_ID"] = self.df["CLASS_ID"].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def _load_frames(self, file_id: str):
        pkl_path = self.mediapipe_dir / f"{file_id}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Missing pkl file: {pkl_path}")
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {pkl_path}: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError(f"Expected list of frames in {pkl_path}, got {type(data)}")
        return data

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_id = row["FILENAME"]
        label = int(row["CLASS_ID"])

        frames = self._load_frames(file_id)
        x = sequence_to_fixed(
            frames,
            t_fixed=self.feature_spec.t_fixed,
            augment=self.augment,
            **self.feature_spec.temporal_augmentations,
        )

        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


def load_label_map(annotations_csv: str) -> dict[int, str]:
    ann = pd.read_csv(annotations_csv)
    required = {"FILENAME", "CLASS_ID", "LABEL"}
    if not required.issubset(ann.columns):
        raise ValueError(f"annotations CSV must contain {required}, got {ann.columns.tolist()}")

    mapping = {}
    for _, row in ann.iterrows():
        mapping[int(row["CLASS_ID"])] = str(row["LABEL"])
    return mapping
