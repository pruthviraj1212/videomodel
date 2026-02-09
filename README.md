# Trash Detector Package

This project is a reusable Python package for detecting trash in videos using YOLOv8.

## Pre-trained Model

This package includes a pre-trained YOLOv8 model (`models/trash_detector_v1.pt`) specifically trained for trash detection. **You don't need to train anything or download additional models** - just install and use!

The model is automatically used by default, so detection works immediately after installation.

## Installation

1.  Clone the repository.
2.  Install the package:

    ```bash
    pip install .
    ```

    Or for development (editable mode):

    ```bash
    pip install -e .
    ```

    Or directly from GitHub (if you host it there):

    ```bash
    pip install git+https://github.com/YOUR_USERNAME/REPO_NAME.git
    ```

## Usage

### Command Line Interface (CLI)

You can use the `trash-detector` command from anywhere in your terminal.

**Detect trash in a video:**

```bash
trash-detector detect --source input_video.mp4 --output output_video.mp4 --conf 0.5
```

**Detect trash in an image:**

```bash
trash-detector detect --source input_image.jpg --output output_image.jpg
```

**Train a model:**

```bash
trash-detector train --data data.yaml --epochs 50 --batch 16
```

### Python API

You can also use the package in your Python scripts.

**Inference:**

```python
from trash_detector import TrashDetector

# Initialize detector
detector = TrashDetector(model_path="yolov8n.pt")

# Process a video
detector.process_video("input_video.mp4", "output_video.mp4")

# Process a single frame
import cv2
frame = cv2.imread("image.jpg")
results = detector.predict_frame(frame)
print(results)
```

**Training:**

```python
from trash_detector import TrashTrainer

# Initialize trainer
trainer = TrashTrainer(model_name="yolov8n.pt")

# Train model
trainer.train(data_yaml_path="data.yaml", epochs=10)
```

## Hardware Acceleration

This package automatically detects and uses Apple Silicon (MPS) or CUDA if available.
