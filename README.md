## Bird Images Acquisition

The dashboard.py file creates a simple web interface using Gradio to show a live webcam feed, detect motion, and display saved images.

### Functions
- `change_detection_stream(dummy=None)`: This function generates a frame with information about the mean optical flow.
- `get_recent_images_gallery(n=10)`: This function retrieves the most recent images from the `data/captured_images` folder.
- `resize_with_aspect_ratio(frame)`: This function resizes a frame while maintaining its aspect ratio.
- `handle_frame(frame)`: This function handles incoming frames from the webcam stream.

### Usage
1. Launch the Gradio UI to view the real-time camera feed and change detection.
2. Adjust the sliders for sensitivity and interval settings.
3. Click the "▶ Start Saving" button to begin periodic image saving.
4. Click the "⏹ Stop Saving" button to stop periodic image saving.
5. Click the "🔄 Refresh Gallery" button to update the captured images gallery.

### Notes
- The dashboard interacts with the `stream_utilis.py` and `image_saver.py` files.
- Images are saved in the `data/captured_images` folder with timestamps.
- Background images are saved every 60 seconds to capture the background.
- Motion-detected images are saved with a "change" prefix.

For more details, refer to the code comments in the `dashboard.py`, `stream_utilis.py`, and `image_saver.py` files.
