# filename: image_saver.py
# author: Lucie 
# date: 2025-07-15
# functions: save_image, start_periodic_image_saving, save_loop, stop_periodic_image_saving
# description: saves pictures from a webcam stream, working with stream_utils.py.
#              It saves two types of pictures: change pictures when it detects movement (using motion data) and "neutral" pictures every 60 seconds to capture the background . 
#              Pictures are saved in the data/captured_images folder with timestamps. The script runs in the background so it doesn’t slow down the video

import os  # for file management
import cv2 # for image processing
import time # for timestamps
from datetime import datetime # for timestamps
import threading # for background saving

# importing frame_stack and FLAGS from stream_utils.py
from stream_utils import frame_stack, FLAGS

SAVE_FOLDER = "data/captured_images" # folder where the all frames will be saved
SAVE_FOLDER_ORIGINAL = SAVE_FOLDER # original folder path to save the frames, used to create subfolders for each camera
os.makedirs(SAVE_FOLDER, exist_ok=True) # create the folder if it doesn't exist

collecting_images = False # flag to indicate if we are currently collecting images or not , initially set to False
BACKGROUND_INTERVAL_SECONDS = 60 # interval in seconds between saving background images
NEUTRAL_FRAMES_TO_SAVE = 10 # number of neutral frames to save when not detecting motion
last_background_save_time = time.time() # time of the last background save
FLAGS["SAVING_NEUTRALS"] = False # flag to indicate if we are currently saving a background image or not
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
def start_periodic_image_saving(interval_seconds=2, 
                                mean_norm=0.4, 
                                # Sometimes, a strangely huge change can be detected (e.g., someone passing by), 
                                # so we set a max threshold for the mean of the flow magnitude normalized.
                                max_mean_norm=.7, #  default is 0.7
                                # interval in seconds between saving background images
                                background_interval_seconds = BACKGROUND_INTERVAL_SECONDS, 
                                save_subfolder = None, # Each camera will save in its own subfolder, if provided.
                                ):
    print(f"interval_seconds: {interval_seconds}")
    global collecting_images, last_background_save_time, background_frame_count
    global SAVE_FOLDER, SAVE_FOLDER_ORIGINAL
    if (save_subfolder is not None) and (len(save_subfolder.strip()) != 0): # if a subfolder is provided, we create it inside the SAVE_FOLDER
        SAVE_FOLDER = os.path.join(SAVE_FOLDER_ORIGINAL, save_subfolder)
        os.makedirs(SAVE_FOLDER, exist_ok=True) # create the folder if it doesn't exist
    else:
        SAVE_FOLDER = SAVE_FOLDER_ORIGINAL
    collecting_images = True # setting the flag to True to start collecting frames

    def save_loop():
        global last_background_save_time, background_frame_count
        FLAGS["SAVING_CHANGES"] = False
        last_change_save_time = 0
# importing flow_magnitude_normalized from stream_utils
        while collecting_images: # looping until the flag is set to False

            if (len(frame_stack) > 1) and not(FLAGS["SAVING_NEUTRALS"]): # checking if there are more than one frames in the frame stack
                from stream_utils import flow_magnitude_normalized # Moved it here to make sure that it's update when accessed.
                current_mean = flow_magnitude_normalized.mean() # getting the mean of the flow magnitude normalized
                # Here, we are checking if the mean is greater than or equal to the threshold 
                # and less than or equal to the max threshold
                if (current_mean > float(mean_norm)) and (current_mean < float(max_mean_norm)): 
                    if not FLAGS["SAVING_CHANGES"]: # checking if the flag is False
                        print(f"[info] Change detected (mean={current_mean:.3f} ≥ threshold={mean_norm}), saving started.")
                        FLAGS["SAVING_CHANGES"] = True
                    if (time.time() - last_change_save_time) > interval_seconds:
                        # We need to save two consecutive images
                        frame_1 = frame_stack.pop() # getting the last frame from the frame stack and removing it
                        frame_0 = frame_stack.pop() # getting the second last frame from the frame stack and removing it 
                        save_image(frame_0, prefix="change_0_")
                        save_image(frame_1, prefix="change_1_")

                        last_change_save_time = time.time()
                else: # if the mean is less than the threshold
                    if FLAGS["SAVING_CHANGES"]: 
                        print(f"[info] No change detected (mean={current_mean:.3f} < threshold={mean_norm}), stopped.")
                    FLAGS["SAVING_CHANGES"] = False


            current_time = time.time()
            if current_time - last_background_save_time >= background_interval_seconds: # checking if the time difference is greater than or equal to the interval  
                
                FLAGS["SAVING_NEUTRALS"] = True # set the flag to true to start saving the background frames
                background_frame_count = 0
                last_background_save_time = current_time
# checking if backgoud saving flag is true , the saved frames will have neutral prefix
            if (len(frame_stack) == 0) and FLAGS["SAVING_NEUTRALS"]:
                print("[DUMMY] STACK EMPTY! ...")
            if FLAGS["SAVING_NEUTRALS"] and (len(frame_stack) > 0):
                frame = frame_stack.pop() # getting the last frame from the frame stack
                print(f"[info] Saving background frame; count: {background_frame_count}")
                save_image(frame, prefix="neutral")
                background_frame_count += 1
                if background_frame_count >= NEUTRAL_FRAMES_TO_SAVE:
                    FLAGS["SAVING_NEUTRALS"] = False
                    print("[info] Finished background frame capture.")

            time.sleep(interval_seconds)

    t = threading.Thread(target=save_loop, daemon=True)
    t.start()
    return f"Saving started (interval={interval_seconds}s, threshold=[{mean_norm}, {max_mean_norm}]) in folder: {SAVE_FOLDER}"

def stop_periodic_image_saving():
    global collecting_images
    collecting_images = False
    return "Image saving stopped"
