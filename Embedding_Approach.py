"""
Embedding approach :

Idea : 
 -> Store the embeddings of the 32 cards including back side of the card
 -> Compere with the embedding values .. 

Stage 1 (detector): YOLO-OBB finds every card (one class: "card"), even when rotated.
Stage 2 (retrieval): each card is warped upright, embedded with DINOv2, and matched
                     against a FAISS index built from one reference image per card.
Stage 3 (temporal):  tracker IDs + a sliding-window vote smooth out per-frame errors.

Reference images: refs/<label>.png  e.g. AS.png, 7S.png, 10H.png, QD.png ... BACK.png
they are stored in the dataset/references folder

Usage:
    pip install torch transformers faiss-cpu opencv-python ultralytics numpy
    python Embedding_Approach.py build --refs dataset/reference --out cards
    python Embedding_Approach.py run --video match.mp4 --detector cards_obb.pt --index cards
"""
import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from transformers import AutoModel
from ultralytics import YOLO

CARD_W, CARD_H = 224, 308        # canonical upright card size (multiples of 14 = DINOv2 patch)
CORNER_W, CORNER_H = 112, 154    # rank+suit corner crop size


# --------------------------------------------------------------------------- #
# Embedding: full card + top-left corner (helps separate e.g. 7 vs 8 of spades)
# --------------------------------------------------------------------------- #
class Embedder:
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, name="facebook/dinov2-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    @torch.no_grad()
    def _embed(self, imgs, size):
        batch = []
        for im in imgs:
            x = cv2.cvtColor(cv2.resize(im, size), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch.append(((x - self.MEAN) / self.STD).transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(batch)).to(self.device)
        out = self.model(pixel_values=x).last_hidden_state[:, 0]  # CLS token
        return torch.nn.functional.normalize(out, dim=-1).cpu().numpy()

    def features(self, cards, bs=64):
        """cards: list of upright BGR crops (any size). Returns L2-normalised float32 [N, 768]."""
        out = []
        for i in range(0, len(cards), bs):
            chunk = [cv2.resize(c, (CARD_W, CARD_H)) for c in cards[i:i + bs]]
            corners = [c[: int(0.30 * CARD_H), : int(0.24 * CARD_W)] for c in chunk]
            f = np.concatenate(
                [self._embed(chunk, (CARD_W, CARD_H)), self._embed(corners, (CORNER_W, CORNER_H))], axis=1
            )
            out.append(f)
        f = np.ascontiguousarray(np.concatenate(out), dtype=np.float32)
        faiss.normalize_L2(f)
        return f


# --------------------------------------------------------------------------- #
# Augmentation of reference images (simulate camera, lighting, glare, tilt)
# --------------------------------------------------------------------------- #
def augment(img, rng):
    h, w = img.shape[:2]
    d = 0.04 * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-d, d, src.shape).astype(np.float32)
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE)
    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-30, 30))
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 2.0))
    if rng.random() < 0.3:  # specular glare blob
        mask = np.zeros((h, w), np.float32)
        cv2.circle(mask, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                   int(rng.uniform(0.1, 0.4) * w), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 25)[..., None]
        out = np.clip(out + mask * rng.uniform(40, 120), 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
# Build FAISS index
# --------------------------------------------------------------------------- #
def build_index(ref_dir, out, n_aug=30):
    emb = Embedder()
    rng = np.random.default_rng(0)
    feats, labels = [], []
    paths = sorted(p for p in Path(ref_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) # though we have only jpg images
    assert paths, f"no reference images in {ref_dir}"
    for p in paths:
        img = cv2.resize(cv2.imread(str(p)), (CARD_W, CARD_H))
        variants = [img] + [augment(img, rng) for _ in range(n_aug)]
        variants += [cv2.rotate(v, cv2.ROTATE_180) for v in variants]  # upside-down copies
        f = emb.features(variants)
        feats.append(f)
        labels += [p.stem] * len(f)
        print(f"{p.stem}: {len(f)} vectors")
    feats = np.concatenate(feats)
    index = faiss.IndexFlatIP(feats.shape[1])  # cosine similarity (vectors are normalised)
    index.add(feats)
    faiss.write_index(index, f"{out}.faiss")
    json.dump(labels, open(f"{out}.json", "w"))
    print(f"saved {index.ntotal} vectors, {len(set(labels))} classes")


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def order_pts(p):
    """4 points clockwise (image coords)"""
    c = p.mean(0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    return np.roll(p, -int(np.argmin(p.sum(1))), axis=0)


def warp_card(frame, pts):
    pts = order_pts(pts.astype(np.float32))
    w = np.linalg.norm(pts[1] - pts[0])
    h = np.linalg.norm(pts[3] - pts[0])
    if w > h:  # make the short edge the first edge -> portrait, no mirroring
        pts = np.roll(pts, -1, axis=0)
    dst = np.float32([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]])
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(frame, M, (CARD_W, CARD_H))


# --------------------------------------------------------------------------- #
# Video inference
# --------------------------------------------------------------------------- #
def run_video(video, detector, index_path, out_video="annotated.mp4", out_json="detections.json",
              window=15, min_score=0.6, conf=0.4):
    det = YOLO(detector)
    emb = Embedder()
    index = faiss.read_index(f"{index_path}.faiss")
    labels = json.load(open(f"{index_path}.json"))

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    history = defaultdict(lambda: deque(maxlen=window))  # track_id -> recent (label, score)
    results = {}

    stream = det.track(source=video, stream=True, persist=True, conf=conf,
                       tracker="bytetrack.yaml", verbose=False)
    for fi, r in enumerate(stream):
        frame = r.orig_img.copy()
        frame_res = {}
        if r.obb is not None and len(r.obb) > 0:
            quads = r.obb.xyxyxy.cpu().numpy().reshape(-1, 4, 2)
            ids = (r.obb.id.int().tolist() if r.obb.id is not None else list(range(len(quads))))
            crops = [warp_card(frame, q) for q in quads]
            D, I = index.search(emb.features(crops), 5)

            for q, tid, d, i in zip(quads, ids, D, I):
                # best similarity per label among the top-k neighbours
                best = {}
                for s, j in zip(d, i):
                    best[labels[j]] = max(best.get(labels[j], -1), float(s))
                lab, score = max(best.items(), key=lambda kv: kv[1])
                if score >= min_score:
                    history[tid].append((lab, score))

                # sliding-window weighted vote
                votes = Counter()
                for l, s in history[tid]:
                    votes[l] += s
                final = votes.most_common(1)[0][0] if votes else "?"
                frame_res[tid] = final

                poly = q.astype(np.int32)
                color = (0, 200, 255) if final == "BACK" else (0, 255, 0)
                cv2.polylines(frame, [poly], True, color, 2)
                cx, cy = poly.mean(0).astype(int)
                cv2.putText(frame, f"{final} {score:.2f}", (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        results[fi] = frame_res
        writer.write(frame)

    writer.release()
    json.dump(results, open(out_json, "w"))
    print(f"wrote {out_video} and {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--refs", default="refs")
    b.add_argument("--out", default="cards")
    b.add_argument("--n_aug", type=int, default=30)

    r = sub.add_parser("run")
    r.add_argument("--video", required=True)
    r.add_argument("--detector", default="cards_obb.pt")
    r.add_argument("--index", default="cards")
    r.add_argument("--window", type=int, default=15)
    r.add_argument("--min_score", type=float, default=0.6)

    a = ap.parse_args()
    if a.cmd == "build":
        build_index(a.refs, a.out, a.n_aug)
    else:
        run_video(a.video, a.detector, a.index, window=a.window, min_score=a.min_score)