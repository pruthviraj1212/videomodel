<<<<<<< HEAD
# videomodel
=======
# Trash Detection Model

This project uses YOLOv8 to detect trash in videos.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Download dataset (if not already done):
    ```bash
    python download_data.py
    ```

3.  Prepare dataset (after download):
    - Ensure data is in `data/` directory.
    - Update `data.yaml` if necessary.

## Training

Train the model using the default configuration:

```bash
python src/train.py
```

## Hardware Acceleration

This project supports Hardware Acceleration on Apple Silicon (M1/M2/M3) using MPS (Metal Performance Shaders).
- The scripts `src/train.py` and `src/inference.py` automatically detect if MPS is available and use it.
- If MPS is not available, it will default to CUDA (if available) or CPU.

## Inference

Run detection on a video:

```bash
python src/inference.py --source input_video.mp4 --output output_video.mp4 --conf 0.5
```
>>>>>>> a9a1f86 (model)
