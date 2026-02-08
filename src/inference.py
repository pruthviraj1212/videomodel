import cv2
from ultralytics import YOLO
import argparse

def process_video(video_path, model_path, output_path="output.mp4", conf=0.5):
    """
    Process a video file, run YOLOv8 object detection, and save the output.
    
    Args:
        video_path (str): Path to input video.
        model_path (str): Path to trained YOLOv8 model weights (.pt).
        output_path (str): Path to save the processed video.
        conf (float): Confidence threshold for detection.
    """
    # Load the YOLOv8 model
    model = YOLO(model_path)
    
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'auto'
    print(f"Running inference on device: {device}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    # 'avc1' (H.264) is generally more compatible with Mac/QuickTime than 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on the frame
            results = model.predict(frame, conf=conf, device=device)

            # Plot the results on the frame
            annotated_frame = results[0].plot()

            # Write the frame to the output video
            out.write(annotated_frame)

            # Display the frame (optional)
            # cv2.imshow('YOLOv8 Inference', annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    except KeyboardInterrupt:
        print("Interrupted by user. Saving current progress...")
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on video.")
    parser.add_argument("--source", required=True, help="Path to input video file.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to model weights.")
    parser.add_argument("--output", default="output.mp4", help="Path to save output video.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    args = parser.parse_args()

    process_video(args.source, args.weights, args.output, args.conf)
