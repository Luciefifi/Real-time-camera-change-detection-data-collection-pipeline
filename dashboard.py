# import gradio as gr
# import glob
# import os
# import cv2
# import numpy as np
# from stream_utilis import frame_stack, compute_optical_flow, video_stream_webcam, FLAGS, flow_magnitude_normalized, WIDTH_STANDARD, HEIGHT_STANDARD
# from image_saver import start_periodic_image_saving, stop_periodic_image_saving

# def change_detection_stream(useless_var=None):
#     """
#     Performs change detection by thresholding optical flow and drawing bounding boxes.
#     """
#     detected_frame = np.zeros((256, 256, 3), dtype=np.uint8) + 127
#     while True:
#         if len(frame_stack) > 1:
#             FLAGS["OBJECT_DETECTING"] = True
#             frame = frame_stack[-1]
#             ret, thresh = cv2.threshold((flow_magnitude_normalized * 255).astype(np.uint8), 127, 255, 0)
#             contours_tuple = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             contours = contours_tuple[0] if len(contours_tuple) == 2 else contours_tuple[1]
#             detected_frame = frame.copy()
#             for contour in contours:
#                 x, y, w, h = cv2.boundingRect(contour)
#                 cv2.rectangle(detected_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             FLAGS["OBJECT_DETECTING"] = False
#         yield detected_frame
#         FLAGS["OBJECT_DETECTING"] = False

# def get_recent_images_gallery(n=10):
#     """
#     Returns the n most recent images from the captured_images folder.
#     """
#     images = sorted(glob.glob("data/captured_images/*.jpg"), key=os.path.getmtime, reverse=True)
#     return images[:n]

# def resize_with_aspect_ratio(frame):
#     h, w = frame.shape[:2]
#     aspect = w / h
#     if w > h:
#         new_w = WIDTH_STANDARD
#         new_h = int(WIDTH_STANDARD / aspect)
#     else:
#         new_h = HEIGHT_STANDARD
#         new_w = int(HEIGHT_STANDARD * aspect)
#     return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

# with gr.Blocks() as demo:
#     gr.Markdown("# 📷 Real-Time Camera Data Collection Pipeline (Change Detection)")

#     # First Row: Webcam Feed and Optical Flow
#     with gr.Row():
#         with gr.Column():
#             webcam_img = gr.Image(label="Webcam", sources="webcam")
#             webcam_img.stream(
#                 fn=lambda s: [frame_stack.append(resize_with_aspect_ratio(s)), print(f"Frame shape: {s.shape}")][0],
#                 inputs=webcam_img,
#                 outputs=None,
#                 time_limit=15,
#                 stream_every=1.0,
#                 concurrency_limit=30
#             )
#         with gr.Column():
#             optical_flow_img = gr.Interface(
#                 compute_optical_flow,
#                 inputs=gr.Slider(label="Optical Flow: Noise Tolerance", minimum=0.0, maximum=1.0, value=0.4),
#                 outputs="image"
#             )

#     # Second Row: Change Detection
#     with gr.Row():
#         with gr.Column():
#             detection_img = gr.Interface(
#                 change_detection_stream,
#                 inputs=gr.Textbox(label="Changes will be detected here", value="DUMMY"),
#                 outputs="image"
#             )

#     # Third Row: Controls and Status
#     with gr.Row():
#         with gr.Column():
#             frame_rate_input = gr.Textbox(label="Frame Rate (FPS)", value="5")
#             save_interval = gr.Slider(minimum=1, maximum=60, value=5, label="Save Image Every (Seconds)")
#             start_btn = gr.Button("▶ Start")
#             stop_btn = gr.Button("⏹ Stop")
#             refresh_btn = gr.Button("🔄 Refresh Image Gallery")
#             status = gr.Textbox(label="Status")

#     # Fourth Row: Gallery
#     with gr.Row():
#         gallery = gr.Gallery(label="Captured Images", show_label=True)

#     # Start streaming and saving
#     start_btn.click(
#         fn=video_stream_webcam,
#         inputs=[frame_rate_input],
#         outputs=webcam_img
#     )

#     start_btn.click(
#         fn=start_periodic_image_saving,
#         inputs=[save_interval, gr.Slider(label="Optical Flow: Noise Tolerance", minimum=0.0, maximum=1.0, value=0.4)],
#         outputs=status
#     )

#     # Stop saving
#     stop_btn.click(
#         fn=stop_periodic_image_saving,
#         outputs=status
#     )

#     # Refresh gallery
#     refresh_btn.click(
#         fn=get_recent_images_gallery,
#         outputs=gallery
#     )

# demo.launch(share=False)

import gradio as gr
import glob
import os
import cv2
import numpy as np

from stream_utilis import frame_stack, compute_optical_flow, WIDTH_STANDARD, HEIGHT_STANDARD
from image_saver import start_periodic_image_saving, stop_periodic_image_saving

from datetime import datetime


def change_detection_stream(dummy=None):
    detected_frame = np.zeros((256, 256, 3), dtype=np.uint8) + 127
    from stream_utilis import flow_magnitude_normalized, FLAGS
    while True:
        if len(frame_stack) > 1:
            FLAGS["OBJECT_DETECTING"] = True
            frame = frame_stack[-1]
            ret, thresh = cv2.threshold((flow_magnitude_normalized * 255).astype(np.uint8), 127, 255, 0)
            contours_tuple = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_tuple[0] if len(contours_tuple) == 2 else contours_tuple[1]
            detected_frame = frame.copy()
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(detected_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            FLAGS["OBJECT_DETECTING"] = False
        yield detected_frame
        FLAGS["OBJECT_DETECTING"] = False

def get_recent_images_gallery(n=10):
    images = sorted(glob.glob("data/captured_images/*.jpg"), key=os.path.getmtime, reverse=True)
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

# Launch Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🛰️ Real-Time Camera (Change Detection)")

    with gr.Row():
        with gr.Column():
            webcam_img = gr.Image(label="📷 Webcam", sources="webcam")
            webcam_img.stream(
                fn=lambda frame: frame_stack.append(resize_with_aspect_ratio(frame)),
                inputs=webcam_img,
                outputs=None,
                time_limit=15,
                stream_every=1.0,
                concurrency_limit=30
            )
        with gr.Column():
            optical_flow_img = gr.Interface(
                compute_optical_flow,
                inputs=gr.Slider(label="Optical Flow: Noise Tolerance", minimum=0.0, maximum=1.0, value=0.4),
                outputs="image"
            )

    with gr.Row():
        with gr.Column():
            detection_img = gr.Interface(
                change_detection_stream,
                inputs=gr.Textbox(label="Change Detection Trigger", value="DUMMY"),
                outputs="image"
            )

    with gr.Row():
        with gr.Column():
            interval_slider = gr.Slider(minimum=1, maximum=60, value=5, label="Save Every (sec)")
            start_button = gr.Button("▶ Start Saving")
            stop_button = gr.Button("⏹ Stop Saving")
            status = gr.Textbox(label="Status", interactive=False)
        with gr.Column():
            refresh_button = gr.Button("🔄 Refresh Gallery")
            image_gallery = gr.Gallery(label="📸 Captured Images", columns=5, rows=2, height=300)

    start_button.click(
        fn=start_periodic_image_saving,
        inputs=[interval_slider],
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

    # Automatically show gallery when app loads
    demo.load(fn=get_recent_images_gallery, outputs=image_gallery)

demo.launch()
