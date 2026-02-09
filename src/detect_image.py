"""
Garbage Detection from Images using YOLOv8

This script detects garbage/trash in images using the trained YOLOv8 model.
"""

import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path
import torch


def detect_garbage(image_path, model_path, output_dir="outputs", conf=0.3, save=True, show=False):
    """
    Detect garbage in an image using YOLOv8.
    
    Args:
        image_path (str): Path to input image or directory of images.
        model_path (str): Path to trained YOLOv8 model weights (.pt).
        output_dir (str): Directory to save detection results.
        conf (float): Confidence threshold for detection.
        save (bool): Whether to save the annotated image.
        show (bool): Whether to display the image (requires GUI).
    
    Returns:
        list: Detection results for each image.
    """
    # Load the YOLOv8 model
    model = YOLO(model_path)
    
    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Running inference on device: {device}")
    
    # Create output directory if saving
    if save:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Handle single image or directory
    image_path = Path(image_path)
    if image_path.is_dir():
        image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.jpeg")) + \
                      list(image_path.glob("*.png")) + list(image_path.glob("*.webp"))
    else:
        image_files = [image_path]
    
    print(f"Processing {len(image_files)} image(s)...")
    
    all_results = []
    
    for img_file in image_files:
        print(f"\nProcessing: {img_file.name}")
        
        # Read image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"Error: Could not read image {img_file}")
            continue
        
        # Run inference
        results = model.predict(image, conf=conf, device=device)
        
        # Get detection details
        detections = results[0].boxes
        num_detections = len(detections)
        
        print(f"  Found {num_detections} garbage detection(s)")
        
        # Print each detection
        for i, box in enumerate(detections):
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls]
            xyxy = box.xyxy[0].tolist()
            print(f"    [{i+1}] {class_name}: {confidence:.2%} at [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
        
        # Get annotated frame
        annotated_image = results[0].plot()
        
        # Save the annotated image
        if save:
            output_file = output_path / f"detected_{img_file.name}"
            cv2.imwrite(str(output_file), annotated_image)
            print(f"  Saved: {output_file}")
        
        # Display if requested
        if show:
            cv2.imshow(f"Garbage Detection - {img_file.name}", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        all_results.append({
            'image': str(img_file),
            'num_detections': num_detections,
            'detections': [
                {
                    'class': model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                }
                for box in detections
            ]
        })
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Detect garbage in images using YOLOv8")
    parser.add_argument("--source", required=True, 
                        help="Path to input image or directory of images")
    parser.add_argument("--weights", 
                        default="runs/detect/models/trash_detector/weights/best.pt",
                        help="Path to model weights (.pt)")
    parser.add_argument("--output", default="outputs", 
                        help="Directory to save output images")
    parser.add_argument("--conf", type=float, default=0.3, 
                        help="Confidence threshold (default: 0.3)")
    parser.add_argument("--show", action="store_true", 
                        help="Display detection results")
    parser.add_argument("--no-save", action="store_true", 
                        help="Don't save annotated images")
    
    args = parser.parse_args()
    
    results = detect_garbage(
        image_path=args.source,
        model_path=args.weights,
        output_dir=args.output,
        conf=args.conf,
        save=not args.no_save,
        show=args.show
    )
    
    # Summary
    total_detections = sum(r['num_detections'] for r in results)
    print(f"\n{'='*50}")
    print(f"Summary: Processed {len(results)} image(s), found {total_detections} total garbage detection(s)")


if __name__ == "__main__":
    main()
