import cv2
import torch
import numpy as np
import time
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import cxy_wh_2_rect


class DaSiamRPNTracker:
    def __init__(self, model_path='models/SiamRPNVOT.model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = SiamRPNvot()
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval().to(self.device)

        self.state = None
        self.last_good_state = None

        # confidence smoothing
        self.score_ema = None
        self.alpha = 0.7

        print("[INFO] DaSiamRPN model loaded on", self.device)

    # -----------------------------------------------------------
    # INIT FROM MASK (SAM / CLIPSeg OUTPUT)
    # -----------------------------------------------------------
    def init_from_mask(self, rgb_frame, mask):
        if rgb_frame is None or mask is None:
            raise ValueError("rgb_frame and mask are required")

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Mask is empty")

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        w = max(10, x_max - x_min)
        h = max(10, y_max - y_min)

        cx = x_min + w / 2
        cy = y_min + h / 2

        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])

        self.state = SiamRPN_init(rgb_frame, target_pos, target_sz, self.net)
        self.last_good_state = self.state.copy()

        print(f"[INFO] Tracker initialized at ({x_min},{y_min},{w},{h})")
        return (x_min, y_min, w, h)

    # -----------------------------------------------------------
    # LIVE TRACKING
    # -----------------------------------------------------------
    def track_live(self, video_src=0, display=True):
        if self.state is None:
            raise RuntimeError("Call init_from_mask() before tracking")

        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video source")

        screen_w, screen_h = 512, 512
        CONF_THRESH = 0.35

        lost_count = 0
        MAX_LOST = 15

        print("[INFO] Live tracking started")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (screen_w, screen_h))
            start = time.time()

            # ---- TRACK ----
            self.state = SiamRPN_track(self.state, frame)

            raw_score = self.state.get("best_score", 1.0)

            # EMA smoothing
            if self.score_ema is None:
                self.score_ema = raw_score
            else:
                self.score_ema = self.alpha * self.score_ema + (1 - self.alpha) * raw_score

            score = self.score_ema

            # bbox
            x, y, w, h = map(
                int,
                cxy_wh_2_rect(self.state['target_pos'], self.state['target_sz'])
            )

            # ---- CLAMP BOUNDARY (NO FALSE LOSS) ----
            x = max(0, min(x, screen_w - w))
            y = max(0, min(y, screen_h - h))

            weak = score < CONF_THRESH

            if weak:
                lost_count += 1
                print(f"[WARN] Weak tracking | score={score:.2f}")

                if self.last_good_state is not None:
                    # freeze size, allow mild position update
                    self.state['target_sz'] = self.last_good_state['target_sz']
                    self.state = self.last_good_state.copy()

                    x, y, w, h = map(
                        int,
                        cxy_wh_2_rect(
                            self.state['target_pos'],
                            self.state['target_sz']
                        )
                    )

            else:
                lost_count = 0
                self.last_good_state = self.state.copy()

            # ---- HARD LOST (V2 RE-DETECTION SLOT) ----
            if lost_count >= MAX_LOST:
                print("[ERROR] Target LOST — holding last known bbox")
                lost_count = MAX_LOST
                yield (x, y, w, h)
                continue

            fps = int(1 / (time.time() - start + 1e-6))

            # ---- DISPLAY ----
            if display:
                color = (0, 255, 0) if not weak else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"FPS:{fps}  Score:{score:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                cv2.imshow("DaSiamRPN Tracker", frame)

            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                break

            yield (x, y, w, h)

        cap.release()
        cv2.destroyAllWindows()
