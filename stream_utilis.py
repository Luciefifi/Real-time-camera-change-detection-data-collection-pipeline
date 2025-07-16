import cv2
import numpy as np
import time
from collections import deque

# Define standard resolution
WIDTH_STANDARD = 640
HEIGHT_STANDARD = 480

# Global variables
frame_stack = deque(maxlen=2)
fall_back_frame = np.zeros((256, 256, 3), dtype=np.uint8) + 127
flow_magnitude_normalized = np.zeros((256, 256), dtype=np.uint8)
FLAGS = {"OBJECT_DETECTING": False}

def compare_images_optical_flow(img1, img2):
    """
    Compares two images and returns a grayscale image of flow magnitude normalized to 0 - 1.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return flow_magnitude

def compute_optical_flow(mean_norm=None):
    """
    Computes optical flow between consecutive frames and yields normalized flow magnitude.
    """
    global FLAGS, flow_magnitude_normalized, frame_stack
    if mean_norm is None:
        mean_norm = 0.4
    else:
        mean_norm = float(mean_norm)
    FLAGS["OBJECT_DETECTING"] = False
    while True:
        if len(frame_stack) > 1 and not FLAGS["OBJECT_DETECTING"]:
            prev_frame, curr_frame = frame_stack
            original_height, original_width = curr_frame.shape[:2]
            prev_frame_resized, curr_frame_resized = [
                cv2.resize(frame, (original_width // 4, original_height // 4))
                for frame in [prev_frame, curr_frame]
            ]
            flow_magnitude = compare_images_optical_flow(prev_frame_resized, curr_frame_resized)
            flow_magnitude_normalized = cv2.normalize(flow_magnitude, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            flow_magnitude_normalized = cv2.resize(
                flow_magnitude_normalized, (original_width, original_height)
            )
            yield flow_magnitude_normalized
        else:
            yield np.stack((flow_magnitude_normalized, flow_magnitude_normalized * 0, flow_magnitude_normalized * 0), axis=-1)

def video_stream_webcam(frame_rate=""):
    """
    Captures frames from the default webcam and adds them to frame_stack.
    """
    if frame_rate.strip() == "":
        frame_rate = 2.0
    else:
        frame_rate = float(frame_rate)
    cap = cv2.VideoCapture(0)  # Default webcam
    if not cap.isOpened():
        print("[error] Failed to open webcam")
        while True:
            yield fall_back_frame
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_stack.append(
                cv2.resize(frame, (WIDTH_STANDARD, HEIGHT_STANDARD))
            )
            yield frame
            time.sleep(1 / frame_rate)
        else:
            yield fall_back_frame