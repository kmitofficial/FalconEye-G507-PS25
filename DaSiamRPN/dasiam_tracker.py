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
        self.net.eval()
        self.net.to(self.device)
        self.state = None
        print("[INFO] DaSiamRPN model loaded successfully on", self.device)

    def init_from_mask(self, rgb_frame, mask):
        if rgb_frame is None or mask is None:
            raise ValueError("Both rgb_frame and mask are required")

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Mask is empty or invalid")

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w, h = x_max - x_min, y_max - y_min
        cx, cy = x_min + w / 2, y_min + h / 2

        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])

        self.state = SiamRPN_init(rgb_frame, target_pos, target_sz, self.net)
        print(f"[INFO] Tracker initialized: x={x_min}, y={y_min}, w={w}, h={h}")
        return (x_min, y_min, w, h)

    def track_live(self, video_src=0, display=True):
        if self.state is None:
            raise RuntimeError("Tracker not initialized. Call init_from_mask() first.")

        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam/video source")

        screen_w, screen_h = 512, 512
        print("[INFO] Live tracking started...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            start = time.time()
            frame = cv2.resize(frame, (screen_w, screen_h))

            self.state = SiamRPN_track(self.state, frame)
            res = cxy_wh_2_rect(self.state['target_pos'], self.state['target_sz'])
            bbox = [int(res[0]), int(res[1]), int(res[2]), int(res[3])]
            end= time.time()
            fps = 1/(end-start)
            cv2.putText(frame,f"FPS: {int(fps)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            # Check if bbox touches any boundary
            x, y, w, h = bbox
            if x <= 0 or y <= 0 or (x + w) >= screen_w or (y + h) >= screen_h:
                print("[INFO] Signal lost — object touched screen boundary.")
                yield (0, 0, 0, 0)
                continue  

            if display:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("DaSiamRPN Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:
                print("[INFO] Tracking stopped by user.")
                break

            yield bbox

        cap.release()
        cv2.destroyAllWindows()
