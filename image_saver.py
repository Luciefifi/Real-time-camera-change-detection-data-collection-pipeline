# filename: image_saver.py
# author: Lucie 
# date: 2025-07-15
# functions: save_image, start_periodic_image_saving, save_loop, stop_periodic_image_saving
# description: saves pictures from a webcam stream, working with stream_utilis.py.
#              It saves two types of pictures: change pictures when it detects movement (using motion data) and "neutral" pictures every 60 seconds to capture the background . 
#              Pictures are saved in the data/captured_images folder with timestamps. The script runs in the background so it doesn’t slow down the video

import os  # for file management
import cv2 # for image processing
import time # for timestamps
from datetime import datetime # for timestamps
import threading # for background saving

# importing frame_stack and FLAGS from stream_utilis.py
from stream_utilis import frame_stack, FLAGS

SAVE_FOLDER = "data/captured_images" # folder where the all frames will be saved
os.makedirs(SAVE_FOLDER, exist_ok=True) # create the folder if it doesn't exist

collecting_images = False # flag to indicate if we are currently collecting images or not , initially set to False
BACKGROUND_INTERVAL_SECONDS = 60 # interval in seconds between saving background images
NEUTRAL_FRAMES_TO_SAVE = 10 # number of neutral frames to save when not detecting motion
last_background_save_time = time.time() # time of the last background save
background_saving = False # flag to indicate if we are currently saving a background image or not
background_frame_count = 0

"""
This function saves a webcam picture (frame) to the data/captured_images folder. It creates a unique file name using a timestamp and a label (change for motion or neutral for background), 
 It converts the picture from RGB to BGR format (needed for saving with OpenCV), saves it as a JPEG, and prints a message showing the file’s location.
 It’s used to store images when motion is detected or for regular background captures.

"""

def save_image(frame, prefix="image"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # creating a timestamp
    filename = os.path.join(SAVE_FOLDER, f"{prefix}_{timestamp}.jpg") # creating a filename
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # converting the frame from RGB to BGR
    cv2.imwrite(filename, frame_bgr) # saving the frame as a JPEG
    print(f"[saved] {filename}")

"""
This function starts a loop that saves a webcam picture (frame) to the data/captured_images folder every 2 seconds. It uses the save_image function to save the frame and sets a flag to indicate that the background is being saved. 
It also sets a counter to keep track of the number of frames saved.

"""
def start_periodic_image_saving(interval_seconds=2, mean_norm=0.4):
    global collecting_images, last_background_save_time, background_saving, background_frame_count
    collecting_images = True # setting the flag to True to start collecting frames

    def save_loop():
        global last_background_save_time, background_saving, background_frame_count
        saving_change_frames = False
        last_change_save_time = 0
# importing flow_magnitude_normalized from stream_utilis
        from stream_utilis import flow_magnitude_normalized

        while collecting_images: # looping until the flag is set to False
            if len(frame_stack) == 0: # checking if the frame stack is empty
                time.sleep(0.1) # waiting for 0.1 seconds before checking again
                continue

            frame = frame_stack[-1] # getting the last frame from the frame stack

            if len(frame_stack) > 1: # checking if there are more than one frames in the frame stack
                current_mean = flow_magnitude_normalized.mean() # getting the mean of the flow magnitude normalized
                if current_mean >= float(mean_norm): # checking if the mean is greater than or equal to the threshold which is mean_norm
                    if not saving_change_frames: # checking if the flag is False
                        print(f"[info] Change detected (mean={current_mean:.3f} ≥ threshold={mean_norm}), saving started.")
                        saving_change_frames = True
                    if time.time() - last_change_save_time >= interval_seconds:
                        save_image(frame, prefix="change")
                        last_change_save_time = time.time()
                else: # if the mean is less than the threshold
                    if saving_change_frames: 
                        print(f"[info] No change detected (mean={current_mean:.3f} < threshold={mean_norm}), stopped.")
                    saving_change_frames = False

            current_time = time.time()
            if current_time - last_background_save_time >= BACKGROUND_INTERVAL_SECONDS: # checking if the time difference is greater than or equal to the interval  
                background_saving = True # set the flag to true to start saving the background frames
                background_frame_count = 0
                last_background_save_time = current_time
# checking if backgoud saving flag is true , the saved frames will have neutral prefix
            if background_saving:
                save_image(frame, prefix="neutral")
                background_frame_count += 1
                if background_frame_count >= NEUTRAL_FRAMES_TO_SAVE:
                    background_saving = False
                    print("[info] Finished background frame capture.")

            time.sleep(0.1)

    t = threading.Thread(target=save_loop, daemon=True)
    t.start()
    return f"Saving started (interval={interval_seconds}s, threshold={mean_norm})"

def stop_periodic_image_saving():
    global collecting_images
    collecting_images = False
    return "Image saving stopped"
