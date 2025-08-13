# filename: dashboard.py
# author: Lucie
# date: 2025-07-15
# description: creates a simple web interface using Gradio to show a live webcam feed, detect motion, and display saved images
import gradio as gr 
import glob
import os
import cv2
import numpy as np
from datetime import datetime

from stream_utils import frame_stack, compute_optical_flow, WIDTH_STANDARD, HEIGHT_STANDARD
from image_saver import start_periodic_image_saving, stop_periodic_image_saving


def change_detection_stream(dummy=None):
    detected_frame = np.zeros((128, 512, 3), dtype=np.uint8) + 127
    from stream_utils import FLAGS
    while True:
        if len(frame_stack) > 1:
            FLAGS["OBJECT_DETECTING"] = True
            from stream_utils import flow_magnitude_normalized
            mean_OF_string = f"Mean Optical Flow: {flow_magnitude_normalized.mean():.4f}"
            detected_frame_new = cv2.putText(detected_frame.copy(), mean_OF_string, (10, 64), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                         (10, 10, 255), 1, cv2.LINE_AA)
        FLAGS["OBJECT_DETECTING"] = False
        yield detected_frame_new


def get_recent_images_gallery(n=10):
    from image_saver import SAVE_FOLDER
    images = sorted(glob.glob(os.path.join(SAVE_FOLDER,"*.jpg")), key=os.path.getmtime, reverse=True)
    return images[:n]


def resize_with_aspect_ratio(frame):
    h, w = frame.shape[:2]
    aspect = w / h
    if w > h:
        new_w = WIDTH_STANDARD
        new_h = int(WIDTH_STANDARD / aspect)
    else:
        new_h = HEIGHT_STANDARD
        new_w = int(HEIGHT_STANDARD * aspect)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def handle_frame(frame):
    from stream_utils import frame_stack
    resized_frame = resize_with_aspect_ratio(frame)
    frame_stack.append(resized_frame)

# Launch Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🛰️ Real-Time Camera (Change Detection)")

    with gr.Row():
        with gr.Column():
            webcam_img = gr.Image(label="📷 Webcam", sources="webcam")   
            webcam_img.stream(
                fn=handle_frame,
                inputs=webcam_img,
                outputs=None,
                time_limit=1,
                stream_every=1.0,
                concurrency_limit=1
            )

        with gr.Column():
            optical_flow_img = gr.Interface(
                compute_optical_flow,
                inputs=gr.Slider(label="Optical Flow: DUMMY PARAMETER", minimum=0.0, maximum=1.0, value=0.4),
                outputs="image"
            )

        with gr.Column():
            detection_img = gr.Interface(
                change_detection_stream,
                inputs=gr.Textbox(label="Change Detection Trigger", value="DUMMY"),
                outputs="image"
            )

    with gr.Row():
        with gr.Column():
            interval_slider = gr.Slider(minimum=0.5, maximum=2, value=1, label="Save Every (sec)")
            background_interval_seconds = gr.Number(label="Background Save Interval (sec)", value=60, precision=0)
            mean_slider = gr.Slider(minimum=0.01, maximum=1.0, value=0.1, label="Change Sensitivity (Optical Flow Mean Threshold)")
            max_slider = gr.Slider(minimum=0.01, maximum=1.0, value=0.7, label="MAX Change torelable (MAX Optical Flow Mean Threshold)")
            save_subfolder = gr.Textbox(label="Save Subfolder (optional)", placeholder="e.g., camera1", value=None)
            start_button = gr.Button("▶ Start Saving")
            stop_button = gr.Button("⏹ Stop Saving")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column():
            refresh_button = gr.Button("🔄 Refresh Gallery")
            image_gallery = gr.Gallery(label="📸 Captured Images", columns=5, rows=2, height=300)

    start_button.click(
        fn=start_periodic_image_saving,
        inputs=[interval_slider, mean_slider, max_slider,
                background_interval_seconds, save_subfolder
                ],
        outputs=status
    )

    stop_button.click(
        fn=stop_periodic_image_saving,
        outputs=status
    )

    refresh_button.click(
        fn=get_recent_images_gallery,
        outputs=image_gallery
    )

    demo.load(fn=get_recent_images_gallery, outputs=image_gallery)

demo.launch()
