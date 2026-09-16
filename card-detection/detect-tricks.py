"""
Detect the sequence of cards played during a single trick ("hand") of 4 turns.

Assumes:
- Camera is fixed, overhead, on the table.
- Players' hand piles are face-down and were NOT part of your training classes,
  so the model will only ever fire detections on face-up PLAYED cards.
- Each turn = one new card appearing face-up on the table; the trick ends when
  the table is cleared (detection count drops back to 0).

This gives you, per trick: the 4 cards played, in the order they were played,
with timestamps.

Usage:
    python detect_trick_turns.py
"""

from collections import defaultdict, Counter
from ultralytics import YOLO
import cv2

# ---------------- CONFIG ----------------
MODEL_PATH = "Z:\\coding\\PR project\\card-detection\\runs\\detect\\train13\\weights\\best.pt"
VIDEO_PATH = "Z:\\coding\\PR project\\29-card-game\\29-gameplay-1 - Converted.mp4"
CONF_THRESHOLD = 0.5

# A track needs to be seen for at least this many frames before we trust its
# class label / count it as a real play (filters out 1-frame false positives).
MIN_TRACK_FRAMES = 5

# Optional: seating positions as (x, y) fractions of frame width/height,
# used only if you want to guess *which player* played each card.
# Fill these in by eyeballing your video (e.g. via the frame you already viewed).
SEATING_POSITIONS = {
    "Player_Top":    (0.5, 0.15),
    "Player_Right":  (0.85, 0.5),
    "Player_Bottom": (0.5, 0.85),
    "Player_Left":   (0.15, 0.5),
}
# -----------------------------------------


def run_tracking(model_path, video_path, conf):
    model = YOLO(model_path)
    class_names = model.names

    results = model.track(
        source=video_path,
        persist=True,
        tracker="bytetrack.yaml",
        conf=conf,
        stream=True,
        verbose=False,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    # track_id -> list of (frame_idx, class_name, cx, cy)
    track_history = defaultdict(list)

    for frame_idx, r in enumerate(results):
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().tolist()
            classes = r.boxes.cls.cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().tolist()
            for track_id, cls_idx, box in zip(ids, classes, xyxy):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                track_history[int(track_id)].append(
                    (frame_idx, class_names[int(cls_idx)], cx, cy)
                )

    return track_history, fps, frame_w, frame_h


def nearest_seat(cx, cy, frame_w, frame_h):
    nx, ny = cx / frame_w, cy / frame_h
    best_seat, best_dist = None, float("inf")
    for seat, (sx, sy) in SEATING_POSITIONS.items():
        dist = (nx - sx) ** 2 + (ny - sy) ** 2
        if dist < best_dist:
            best_dist, best_seat = dist, seat
    return best_seat


def summarize_plays(track_history, fps, frame_w, frame_h):
    plays = []
    for track_id, records in track_history.items():
        if len(records) < MIN_TRACK_FRAMES:
            continue  # likely a flicker / false positive, not a real play

        first_frame = records[0][0]
        class_votes = Counter(r[1] for r in records)
        best_class = class_votes.most_common(1)[0][0]

        avg_cx = sum(r[2] for r in records) / len(records)
        avg_cy = sum(r[3] for r in records) / len(records)
        seat = nearest_seat(avg_cx, avg_cy, frame_w, frame_h)

        plays.append({
            "track_id": track_id,
            "card": best_class,
            "first_seen_sec": first_frame / fps,
            "seat_guess": seat,
        })

    # Order by when each card first appeared -> turn order
    plays.sort(key=lambda p: p["first_seen_sec"])
    return plays


def main():
    track_history, fps, frame_w, frame_h = run_tracking(MODEL_PATH, VIDEO_PATH, CONF_THRESHOLD)
    plays = summarize_plays(track_history, fps, frame_w, frame_h)

    print(f"Detected {len(plays)} played card(s) in this trick (expected 4)\n")
    for turn_num, p in enumerate(plays, start=1):
        print(f"Turn {turn_num}: {p['card']}  "
              f"(played at ~{p['first_seen_sec']:.1f}s, seat guess: {p['seat_guess']})")

    if len(plays) != 4:
        print("\nCount mismatch — check CONF_THRESHOLD / MIN_TRACK_FRAMES, or inspect "
              "whether any card was mis-tracked as two separate track_ids (common if "
              "a card gets briefly occluded by a hand mid-play).")


if __name__ == "__main__":
    main()