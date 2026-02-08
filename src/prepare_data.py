import shutil
import yaml
import random
import os
from pathlib import Path
from tqdm import tqdm

def prepare_dataset(source_path, target_dir="data", split_ratio=(0.8, 0.1, 0.1)):
    """
    Organizes the dataset into YOLOv8 format (train/val/test).
    
    Args:
        source_path (str): Path to the downloaded dataset.
        target_dir (str): Path to the target directory.
        split_ratio (tuple): Train, validation, test split ratios.
    """
    source_path = Path(source_path)
    target_dir = Path(target_dir)
    
    if not source_path.exists():
        print(f"Error: Source path {source_path} does not exist.")
        return

    # Create directories
    for split in ['train', 'val', 'test']:
        (target_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (target_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Collect all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    
    # Recursively find images, assuming labels share the same filename stem
    for root, _, files in os.walk(source_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                images.append(os.path.join(root, file))

    random.shuffle(images)
    num_images = len(images)
    print(f"Found {num_images} images.")
    
    if num_images == 0:
        print("No images found. Please check the source path.")
        return

    train_end = int(num_images * split_ratio[0])
    val_end = train_end + int(num_images * split_ratio[1])

    # Move files
    for i, image_path in enumerate(tqdm(images, desc="Processing images")):
        image_path = Path(image_path)
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'
            
        # Determine label path (assume same name, different ext, usually .txt)
        label_path = image_path.with_suffix('.txt')
        
        # Check if label exists alongside image
        if not label_path.exists():
             print(f"Warning: Label for {image_path.name} not found at {label_path}")
             continue

        # Copy image
        shutil.copy(image_path, target_dir / 'images' / split / image_path.name)
        
        # Process and copy label (Flatten to single class '0')
        new_label_path = target_dir / 'labels' / split / label_path.name
        with open(label_path, 'r') as f_in, open(new_label_path, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Replace class_id (first part) with '0'
                    parts[0] = '0'
                    f_out.write(" ".join(parts) + "\n")

    print(f"Dataset organization complete. Processed {num_images} images.")
    
    # Check if labels were actually created
    label_count = sum(1 for _ in (target_dir / 'labels').rglob('*.txt'))
    if label_count == 0:
        print("\nWARNING: No label files were created! This might be a classification dataset or use a different label format (e.g. XML).")
        print("Please check the source dataset structure.")
    else:
        print(f"Created {label_count} label files.")

    # Create data.yaml
    create_data_yaml(target_dir)

def create_data_yaml(target_dir):
    """
    Creates data.yaml file for YOLOv8.
    """
    # Define classes - this typically requires knowing the dataset. 
    # Since we can't inspect it yet, we'll assume a single 'trash' class or generic classes.
    # User asked for "trash or pile of garbage" - implies single class detection might be best,
    # or we map all classes to 'trash'. 
    
    # For this specific dataset "hammadarshad18/garbage-detection", it likely has multiple classes.
    # We will set it up to auto-detect classes if possible (hard without reading a classes.txt),
    # OR we default to a generic setup and ask user to verify.
    
    # We'll use a placeholder and instruct user to update.
    data = {
        'path': str(target_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'trash' # Placeholder, user needs to check dataset classes
        }
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data, f)
    
    print("Created data.yaml. PLEASE VERIFY CLASS NAMES.")

if __name__ == "__main__":
    # You can update this path manually or pass it as an arg
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", help="Path to the downloaded dataset")
    args = parser.parse_args()
    
    prepare_dataset(args.source_path)
