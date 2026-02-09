import cv2
from ultralytics import YOLO
import torch
import os

class TrashDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the TrashDetector with a trained YOLOv8 model.
        
        Args:
            model_path (str): Path to the trained model weights.
        """
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.device = 'mps' if torch.backends.mps.is_available() else 'auto'
        print(f"TrashDetector initialized on device: {self.device}")

    def process_video(self, video_path, output_path="output.mp4", conf=0.5):
        """
        Process a video file, run detection, and save the output.
        
        Args:
            video_path (str): Path to input video.
            output_path (str): Path to save the processed video.
            conf (float): Confidence threshold.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {video_path} -> {output_path}")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(frame, conf=conf, device=self.device, verbose=False)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

        except KeyboardInterrupt:
            print("Interrupted by user. Saving current progress...")
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"Processed video saved to {output_path}")

    def process_image(self, image_path, output_path="output.jpg", conf=0.5):
        """
        Process a single image file, run detection, and save the output.

        Args:
            image_path (str): Path to input image.
            output_path (str): Path to save the processed image.
            conf (float): Confidence threshold.
        """
        import cv2
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not open image {image_path}")
            return

        print(f"Processing image: {image_path} -> {output_path}")
        results = self.model.predict(frame, conf=conf, device=self.device, verbose=False)
        
        # Print detection count
        for r in results:
            print(f"Detected {len(r.boxes)} objects.")

        annotated_frame = results[0].plot()
        cv2.imwrite(output_path, annotated_frame)
        print(f"Processed image saved to {output_path}")

    def predict_frame(self, frame, conf=0.5):
        """
        Run detection on a single frame.
        
        Args:
            frame (numpy.ndarray): Input image frame.
            conf (float): Confidence threshold.
            
        Returns:
            list: List of detections or the results object.
        """
        results = self.model.predict(frame, conf=conf, device=self.device, verbose=False)
        return results
