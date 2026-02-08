from ultralytics import YOLO

def train_model(data_yaml_path, epochs=50, imgsz=640, batch=16):
    """
    Train a YOLOv8 model on the custom dataset.
    
    Args:
        data_yaml_path (str): Path to the data.yaml file.
        epochs (int): Number of training epochs.
        imgsz (int): Image size for training.
        batch (int): Batch size.
    """
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (nano for speed)

    # Train the model
    # Use 'mps' for Apple Silicon if available, else 'cpu' or 'cuda' (auto)
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'auto'
    print(f"Training on device: {device}")
    
    results = model.train(data=data_yaml_path, epochs=epochs, imgsz=imgsz, batch=batch, project="models", name="trash_detector", device=device)
    
    return results

if __name__ == "__main__":
    try:
        # This path will be updated once we confirm the dataset location
        DATA_YAML = "data.yaml" 
        train_model(DATA_YAML)
    except Exception as e:
        print(f"Error during training: {e}")
