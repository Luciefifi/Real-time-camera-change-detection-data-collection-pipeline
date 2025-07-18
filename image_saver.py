import os
import cv2
import time
from datetime import datetime
import threading

from stream_utilis import frame_stack, FLAGS

SAVE_FOLDER = "data/captured_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

collecting_images = False
BACKGROUND_INTERVAL_SECONDS = 60
NEUTRAL_FRAMES_TO_SAVE = 10
last_background_save_time = time.time()
background_saving = False
background_frame_count = 0

# Shared magnitude placeholder (importable and updated by compute_optical_flow)
last_flow_magnitude = [0.0]  # Wrap in list to be mutable

def save_image(frame, prefix="image"):
    """
    Saves a frame as a JPEG with a timestamped filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(SAVE_FOLDER, f"{prefix}_{timestamp}.jpg")
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, frame_bgr)
    print(f"[saved] {filename}")

def start_periodic_image_saving(interval_seconds=2, mean_norm=0.4):
    """
    Starts a background thread to save images based on change detection and periodic intervals.
    """
    global collecting_images, last_background_save_time, background_saving, background_frame_count
    collecting_images = True

    def save_loop():
        global last_background_save_time, background_saving, background_frame_count
        saving_change_frames = False
        last_change_save_time = 0

        from stream_utilis import flow_magnitude_normalized

        while collecting_images:
            if len(frame_stack) == 0:
                time.sleep(0.1)
                continue

            frame = frame_stack[-1]

            # --- Change-Triggered Saving ---
            if len(frame_stack) > 1:
                current_mean = flow_magnitude_normalized.mean()
                # print(f"[debug] Optical flow mean: {current_mean:.4f}")

                if current_mean >= float(mean_norm):
                    if not saving_change_frames:
                        print("[info] Change detected, starting to save change frames.")
                        saving_change_frames = True
                    if time.time() - last_change_save_time >= interval_seconds:
                        save_image(frame, prefix="change")
                        last_change_save_time = time.time()
                else:
                    if saving_change_frames:
                        print("[info] No change detected, stopping change save.")
                    saving_change_frames = False

            # --- Neutral Frame Saving ---
            current_time = time.time()
            if current_time - last_background_save_time >= BACKGROUND_INTERVAL_SECONDS:
                background_saving = True
                background_frame_count = 0
                last_background_save_time = current_time

            if background_saving:
                save_image(frame, prefix="neutral")
                background_frame_count += 1
                if background_frame_count >= NEUTRAL_FRAMES_TO_SAVE:
                    background_saving = False
                    print("[info] Finished background frame capture.")

            time.sleep(0.1)

    t = threading.Thread(target=save_loop, daemon=True)
    t.start()
    return "Change-based and timed image saving started"

def stop_periodic_image_saving():
    """
    Stops the image saving process.
    """
    global collecting_images
    collecting_images = False
    return "Image saving stopped"
