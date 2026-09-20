"""
Use a Roboflow model (Universe or your own) as the card locator, then identify
each card with the FAISS index built by card_pipeline.py.

    pip install inference supervision faiss-cpu opencv-python torch transformers
    python roboflow.py --video match.mp4 --model_id project-name/1 --index cards

Put this file next to card_pipeline.py.
"""

# Depends on Embedding_approach.py
# Runs based on the embedded value which has already been stored
# first run the Embedding_approach.py to get the results


import argparse
import json
import os
from collections import Counter, defaultdict, deque

import cv2
import faiss
import numpy as np
import supervision as sv
from inference import get_model

from Embedding_Approach import Embedder, warp_card


def quads_from_detections(det):
    """One 4-point quad per detection.
    - Segmentation model: minimum-area rotated rectangle around the mask (handles rotation).
    - Plain detection model: falls back to the axis-aligned box (rotated cards will warp poorly).
    """
    quads = []
    for k, (x1, y1, x2, y2) in enumerate(det.xyxy):
        quad = None
        if det.mask is not None:
            cnts, _ = cv2.findContours(det.mask[k].astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                quad = cv2.boxPoints(cv2.minAreaRect(max(cnts, key=cv2.contourArea)))
        if quad is None:
            quad = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        quads.append(np.float32(quad))
    return quads


def run(video, model_id, index_path, out="annotated_rf.mp4", out_json="detections_rf.json",
        conf=0.4, window=15, min_score=0.6):
    model = get_model(model_id=model_id, api_key=os.environ["ROBOFLOW_API_KEY"])
    emb = Embedder()
    index = faiss.read_index(f"{index_path}.faiss")
    labels = json.load(open(f"{index_path}.json"))

    info = sv.VideoInfo.from_video_path(video)
    tracker = sv.ByteTrack(frame_rate=int(info.fps))
    history = defaultdict(lambda: deque(maxlen=window))
    results = {}

    with sv.VideoSink(out, info) as sink:
        for fi, frame in enumerate(sv.get_video_frames_generator(video)):
            det = sv.Detections.from_inference(model.infer(frame, confidence=conf)[0])
            det = tracker.update_with_detections(det)
            frame_res = {}

            if len(det):
                quads = quads_from_detections(det)
                D, I = index.search(emb.features([warp_card(frame, q) for q in quads]), 5)

                for q, tid, d, i in zip(quads, det.tracker_id, D, I):
                    best = {}
                    for s, j in zip(d, i):
                        best[labels[j]] = max(best.get(labels[j], -1), float(s))
                    lab, score = max(best.items(), key=lambda kv: kv[1])
                    if score >= min_score:
                        history[int(tid)].append((lab, score))

                    votes = Counter()
                    for l, s in history[int(tid)]:
                        votes[l] += s
                    final = votes.most_common(1)[0][0] if votes else "?"
                    frame_res[int(tid)] = final

                    poly = q.astype(np.int32)
                    cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
                    cx, cy = poly.mean(0).astype(int)
                    cv2.putText(frame, f"{final} {score:.2f}", (cx - 30, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            results[fi] = frame_res
            sink.write_frame(frame)

    json.dump(results, open(out_json, "w"))
    print(f"wrote {out} and {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--index", default="cards")
    ap.add_argument("--conf", type=float, default=0.4)
    a = ap.parse_args()
    run(a.video, a.model_id, a.index, conf=a.conf)