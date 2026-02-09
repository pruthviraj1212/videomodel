import cv2
import os

# Load your video model
from model import VideoModel  # Replace with the actual import path of your model
from model import ImageModel  # Replace with the actual import path of your image model

def process_video(video_path):
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    # Initialize the video model
    model = VideoModel()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    print(f"Processing video: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with the model
        result = model.process_frame(frame)  # Replace with your model's method
        print(result)  # Replace with your desired output handling

    cap.release()
    print("Video processing complete.")

def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # Initialize the image model
    model = ImageModel()

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    print(f"Processing image: {image_path}")
    # Process the image with the model
    result = model.process_image(image)  # Replace with your model's method
    print(result)  # Replace with your desired output handling

    print("Image processing complete.")

if __name__ == "__main__":
    # Example usage
    video_path = "/Users/pruthviraj/Downloads/publicpulse/_Garbage_city_in_Cairo_garbagecity_garbage_cairo_egypt_cairo_egypt_tiktokegypt_720P.mp4"
    process_video(video_path)

    image_path = "/Users/pruthviraj/Downloads/download (3).jpeg"
    process_image(image_path)
