from ultralytics import YOLO
import os

def resume_training():
    # Path to the last checkpoint
    # Based on our checks: runs/detect/models/trash_detector/weights/last.pt
    # Note: 'runs/detect' seems to be where YOLO put it despite project="models" arg? 
    # Or maybe 'models' argument was treated as subdirectory of 'runs/detect'.
    
    checkpoint_path = "runs/detect/models/trash_detector/weights/last.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please check where 'last.pt' is located.")
        return

    print(f"Resuming training from {checkpoint_path}...")
    
    # Load the model
    model = YOLO(checkpoint_path)

    # Resume training
    # We don't need to specify data/epochs again, it's saved in the pt file
    model.train(resume=True)

if __name__ == "__main__":
    resume_training()
