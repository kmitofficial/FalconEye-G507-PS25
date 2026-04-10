import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time
from PIL import Image
import torch
import cv2
import numpy as np

from sam_model import segment_on_click
from image_preprocessing import preprocess_frame
from control import RoverController
from boundingbox import get_boundary
from DaSiamRPN.dasiam_tracker import DaSiamRPNTracker
from clip import clipping


class FalconEye(Node):
    def __init__(self):
        super().__init__('falconeye')

        self.sub = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.callback,
            10
        )

        self.frame = None
        self.processed = False  # run pipeline only once

        cv2.namedWindow("FalconEye")
        cv2.setMouseCallback("FalconEye", self.mouse_callback)

    def callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        self.frame = frame
        cv2.imshow("FalconEye", frame)
        cv2.waitKey(1)

        # run main pipeline once after first frame
        if not self.processed:
            self.run_pipeline(frame)
            self.processed = True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at: {x}, {y}")

    def run_pipeline(self, frame):
        print("Capturing frame from ROS...")

        rgb_frame, frame_resized = preprocess_frame(frame)

        print("Choose an option:")
        print("1. Click")
        print("2. Reference image")
        print("3. Text")

        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == '1':
            clicks = int(input("Enter number of clicks: "))
            mask = segment_on_click(rgb_frame, clicks)

        elif choice == '2':
            Tk().withdraw()
            image_path = askopenfilename(
                title="Select image",
                filetypes=[("Image files", "*.jpg *.png")]
            )
            ref_image = Image.open(image_path).convert("RGB")
            ref_image = np.array(ref_image)
            ref_image = cv2.resize(ref_image, (512, 512))
            mask = clipping(rgb_frame, ref_image=ref_image)

        elif choice == '3':
            text_prompt = input("Enter text prompt: ")
            mask = clipping(rgb_frame, text=text_prompt)

        else:
            print("Invalid choice")
            return

        bbox, frame_with_box = get_boundary(mask, frame_resized)

        if bbox:
            print("Bounding box:", bbox)
            cv2.imshow("Tracked Object", frame_with_box)
            cv2.waitKey(1)

        tracker = DaSiamRPNTracker()
        bbox = tracker.init_from_mask(rgb_frame, mask)

        print("[INFO] Initialized tracker:", bbox)

        try:
            for box in tracker.track_live(video_src=0, display=True):
                print("BBox:", box)

        except KeyboardInterrupt:
            print("Stopped")

        finally:
            cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = FalconEye()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()