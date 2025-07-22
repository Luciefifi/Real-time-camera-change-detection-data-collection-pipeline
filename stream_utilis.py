# file: stream_utilis.py
# author: Lucie 
# date: 2025-07-15
# functions : compare_image_optical_flow , compute_optical_flow, video_stream_webcam
# description : this file contains codes that captures video from a webcome, resize frames to a standard resolution (640*480) and computes optical flow(the motion flow between two frames) uwsing franeback method
#               it maintains a queue of two frames to compare them and generates a normalized grayscale image for representing motion intensity


import cv2 # for video capture and frame processing
import numpy as np # for numerical operations
import time # for time measurement
from collections import deque # for maintaining a queue of frames

# Define standard resolution
WIDTH_STANDARD = 640
HEIGHT_STANDARD = 480

# Global variables
frame_stack = deque(maxlen=2)
fall_back_frame = np.zeros((HEIGHT_STANDARD, WIDTH_STANDARD, 3), dtype=np.uint8) + 127
flow_magnitude_normalized = np.zeros((HEIGHT_STANDARD, WIDTH_STANDARD), dtype=np.float32)
FLAGS = {"OBJECT_DETECTING": False}

def compare_images_optical_flow(img1, img2):
    """
    Compares two images and returns a grayscale image of flow magnitude.
    computes optical flow between two RGB images (img1, img2) and returns a grayscale image representing the magnitude motion
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

    FLAGS["OBJECT_DETECTING"] = False # disables object detection mode to allow optical flow computation

    while True:
        if len(frame_stack) > 1 and not FLAGS["OBJECT_DETECTING"]:
            prev_frame, curr_frame = frame_stack # getting the previous and current frames from frame stack

            # Resize both current and previous frame to smaller resolution for faster optical flow 
            prev_resized = cv2.resize(prev_frame, (WIDTH_STANDARD // 4, HEIGHT_STANDARD // 4))
            curr_resized = cv2.resize(curr_frame, (WIDTH_STANDARD // 4, HEIGHT_STANDARD // 4))

            # Compute flow magnitude
            flow_magnitude = compare_images_optical_flow(prev_resized, curr_resized)

            # Normalize to [0, 1]
            flow_magnitude_normalized_local = cv2.normalize(flow_magnitude, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

            # Resize back to original
            flow_magnitude_normalized = cv2.resize(flow_magnitude_normalized_local, (WIDTH_STANDARD, HEIGHT_STANDARD))

            # DEBUG: Show mean of flow
            # print(f"[debug] Optical flow mean: {flow_magnitude_normalized.mean():.4f}")

            # Yield 3-channel grayscale (for Gradio)
            yield np.stack([flow_magnitude_normalized] * 3, axis=-1)

        else:
            # Yield fallback (3-channel zero image)
            yield np.stack([flow_magnitude_normalized] * 3, axis=-1)

def video_stream_webcam(frame_rate=""):
    """
    Captures frames from the default webcam and adds them to frame_stack.
    """
    if frame_rate.strip() == "":
        frame_rate = 2.0
    else:
        frame_rate = float(frame_rate)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Failed to open webcam")
        while True:
            yield fall_back_frame

    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame, (WIDTH_STANDARD, HEIGHT_STANDARD))
            frame_stack.append(resized_frame)
            yield resized_frame
            time.sleep(1 / frame_rate)
        else:
            yield fall_back_frame
