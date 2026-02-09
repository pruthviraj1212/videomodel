import argparse
import sys
from .detector import TrashDetector
from .trainer import TrashTrainer

def main():
    parser = argparse.ArgumentParser(description="Trash Detector CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Detect Command
    detect_parser = subparsers.add_parser("detect", help="Run detection on a video or image")
    detect_parser.add_argument("--source", required=True, help="Path to input video or image")
    detect_parser.add_argument("--output", default="output.mp4", help="Path to save output")
    detect_parser.add_argument("--model", default="models/trash_detector_v1.pt", help="Path to model weights")
    detect_parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", required=True, help="Path to data.yaml")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_parser.add_argument("--base-model", default="yolov8n.pt", help="Base model to start from")

    args = parser.parse_args()

    if args.command == "detect":
        detector = TrashDetector(model_path=args.model)
        
        # Check if source is image or video based on extension
        source_lower = args.source.lower()
        is_image = source_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        
        if is_image:
            if args.output == "output.mp4": # Default value
                base_name = args.source.rsplit('.', 1)[0]
                args.output = f"{base_name}_detected.jpg"
            detector.process_image(args.source, args.output, args.conf)
        else:
            detector.process_video(args.source, args.output, args.conf)
    elif args.command == "train":
        trainer = TrashTrainer(model_name=args.base_model)
        trainer.train(args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
