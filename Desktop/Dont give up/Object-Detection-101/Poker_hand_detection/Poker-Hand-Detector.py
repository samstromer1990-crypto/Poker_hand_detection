import os
from collections import Counter, deque

import cv2
import torch
from ultralytics import YOLO

import PokerHandFunction

CONF_THRESHOLD = 0.5     
IMG_SIZE = 960           
HISTORY = 15              
MIN_VOTES = 8             

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "playingCards.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "poker_hand_result.mp4")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find the model: {MODEL_PATH}")

device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device)

cap = None
for idx in range(3):
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        print("Camera found at index:", idx)
        break
    cap.release()
    cap = None
if cap is None:
    raise RuntimeError("No webcam found.")

cap.set(3, 1280)
cap.set(4, 720)

model = YOLO(MODEL_PATH)
classNames = model.names  


def draw_banner(img, text, sub_text, ok):
    """Centered result banner at the top of the frame."""
    h, w = img.shape[:2]
    banner_h = 110
    overlay = img.copy()
    color = (0, 120, 0) if ok else (60, 60, 60)
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.8, 3)
    cv2.putText(img, text, ((w - tw) // 2, 55), cv2.FONT_HERSHEY_DUPLEX,
                1.8, (255, 255, 255), 3, cv2.LINE_AA)

    (sw, sh), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(img, sub_text, ((w - sw) // 2, 92), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (220, 220, 220), 2, cv2.LINE_AA)


history = deque(maxlen=HISTORY)
writer = None

try:
    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Could not read from webcam.")
            break

        results = model(img, conf=CONF_THRESHOLD, iou=0.5, agnostic_nms=True,
                        imgsz=IMG_SIZE, device=device, verbose=False)

        # One entry per card: keep the best confidence
        # (cards have the label in 2 corners -> duplicates are merged here)
        frame_cards = {}
        for box in results[0].boxes:
            name = classNames[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{name} {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if name not in frame_cards or conf > frame_cards[name]:
                frame_cards[name] = conf

        history.append(set(frame_cards))

        # Vote across recent frames: a card only counts if it was seen
        # consistently, so single-frame misreads can't change the result.
        votes = Counter(c for frame in history for c in frame)
        stable = [c for c, v in votes.most_common() if v >= MIN_VOTES][:5]

        if len(stable) == 5:
            hand_name = PokerHandFunction.findPokerHand(stable)
            draw_banner(img, hand_name, "  ".join(sorted(stable)), ok=True)
        else:
            draw_banner(img, "Show 5 cards",
                        f"Stable cards: {len(stable)}/5   {'  '.join(sorted(stable))}",
                        ok=False)

        if writer is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 20.0
            height, width = img.shape[:2]
            writer = cv2.VideoWriter(
                OUTPUT_PATH,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {OUTPUT_PATH}")

        writer.write(img)
        cv2.imshow("Poker Hand Detector", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            history.clear()  # reset when you change the hand

finally:
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if writer is not None:
    print("Saved result video:", OUTPUT_PATH)
