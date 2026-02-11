#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.data.features import FeatureSpec, frame_to_feature
from src.models.tcn import TCNClassifier, GRUBaseline


def build_runtime_frame_dict(results_pose, results_hands):
    return {
        "pose": {"pose_landmarks": [results_pose.pose_landmarks.landmark] if results_pose.pose_landmarks else []},
        "hands": {
            "hand_landmarks": [hl.landmark for hl in results_hands.multi_hand_landmarks] if results_hands.multi_hand_landmarks else [],
            "handedness": [h.classification for h in results_hands.multi_handedness] if results_hands.multi_handedness else [],
        },
        "holistic_legacy": None,
    }


def overlay_text(frame, lines):
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--feature_spec", type=str, default="artifacts/feature_spec.json")
    parser.add_argument("--label_map", type=str, default="artifacts/label_map.json")
    parser.add_argument("--model", choices=["tcn", "gru"], default="tcn")
    parser.add_argument("--infer_every", type=int, default=2)
    parser.add_argument("--smooth_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    spec = FeatureSpec.from_json(args.feature_spec)
    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "tcn":
        model = TCNClassifier(input_dim=spec.d_frame, num_classes=300)
    else:
        model = GRUBaseline(input_dim=spec.d_frame, num_classes=300)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    buffer = deque(maxlen=spec.t_fixed)
    logits_hist = deque(maxlen=max(1, args.smooth_k))

    frame_idx = 0
    last_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb)
            results_hands = hands.process(rgb)

            runtime_frame = build_runtime_frame_dict(results_pose, results_hands)
            feat = frame_to_feature(runtime_frame)
            buffer.append(feat)

            pred_label, pred_conf = "Calibrando…", 0.0
            if len(buffer) >= spec.t_fixed and frame_idx % max(1, args.infer_every) == 0:
                x = np.stack(buffer, axis=0).astype(np.float32)
                xt = torch.from_numpy(x).unsqueeze(0).to(device)
                mask = (xt.abs().sum(dim=-1) > 0).float()
                with torch.no_grad():
                    logits = model(xt, mask=mask).squeeze(0)
                logits_hist.append(logits.cpu().numpy())

                avg_logits = np.mean(np.stack(logits_hist, axis=0), axis=0)
                probs = torch.softmax(torch.from_numpy(avg_logits), dim=0).numpy()
                cls = int(np.argmax(probs))
                pred_conf = float(probs[cls])
                if pred_conf >= args.threshold:
                    pred_label = label_map.get(str(cls), f"CLASS_{cls}")
                else:
                    pred_label = "Desconocido"

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last_time))
            last_time = now

            lines = [
                f"Pred: {pred_label}",
                f"Conf: {pred_conf:.2f}",
                f"FPS: {fps:.1f}",
                f"Buffer: {len(buffer)}/{spec.t_fixed}",
            ]
            overlay_text(frame, lines)

            cv2.imshow("LSE Realtime", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        hands.close()


if __name__ == "__main__":
    main()
