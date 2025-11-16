import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

sam_checkpoint = "models/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
predictor = SamPredictor(sam)

def call_sam(frame: np.ndarray,box):
    predictor.set_image(frame)
    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            box=box[None, :],   # shape [1, 4]
            multimask_output=False
        )
    best_mask = masks[0].astype(np.uint8)
    return best_mask

# --- Segment on click ---
def segment_on_click(frame_rgb, max_clicks):
    coords = []
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    clone = frame_bgr.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(coords) < max_clicks:
            coords.append((x, y))
            cv2.circle(param, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("Click to Segment", param)

    cv2.imshow("Click to Segment", clone)
    cv2.setMouseCallback("Click to Segment", click_event, clone)

    while True:
        cv2.imshow("Click to Segment", clone)
        key = cv2.waitKey(1) & 0xFF

        if len(coords) == max_clicks:
            break

    cv2.destroyAllWindows()

    # --- Feed RGB frame to SAM ---
    predictor.set_image(frame_rgb)
    point_labels = np.ones(len(coords), dtype=int)

    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords=np.array(coords),
            point_labels=point_labels,
            multimask_output=False
        )

    best_mask = masks[0].astype(np.uint8)
    return best_mask
