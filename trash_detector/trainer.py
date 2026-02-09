from ultralytics import YOLO
import torch

class TrashTrainer:
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initialize the trainer.
        
        Args:
            model_name (str): Base model to start training from.
        """
        self.model = YOLO(model_name)
        self.device = 'mps' if torch.backends.mps.is_available() else 'auto'
        print(f"TrashTrainer initialized on device: {self.device}")

    def train(self, data_yaml_path, epochs=50, imgsz=640, batch=16, project="models", name="trash_detector"):
        """
        Train the model.
        
        Args:
            data_yaml_path (str): Path to data.yaml.
            epochs (int): Number of epochs.
            imgsz (int): Image size.
            batch (int): Batch size.
            project (str): Project name for saving results.
            name (str): specific run name.
        """
        print(f"Starting training with data={data_yaml_path}, epochs={epochs}, device={self.device}")
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            device=self.device
        )
        return results
