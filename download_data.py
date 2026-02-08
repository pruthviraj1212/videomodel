import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("hammadarshad18/garbage-detection")

print("Dataset downloaded to:", path)

# Define target directory
target_dir = os.path.join(os.getcwd(), "data")

# Copy/Move logic could go here, but first let's just see what we got
# create data dir if not exists
os.makedirs(target_dir, exist_ok=True)

print(f"Please inspect {path} and move relevant files to {target_dir}")
