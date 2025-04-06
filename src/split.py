import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = r"D:\CADT\CapstoneProjectI\ml__model\data\raw_images"
output_dir = r"D:\CADT\CapstoneProjectI\ml__model\data\splited_data"

# Define split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output directories
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

# Loop through each class folder
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        # Get list of images in the class folder
        images = [img for img in os.listdir(class_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split the images into train, val, and test sets
        train_images, test_images = train_test_split(images, test_size=test_ratio, random_state=42)
        train_images, val_images = train_test_split(train_images, test_size=val_ratio/(train_ratio + val_ratio), random_state=42)
        
        # Create class subdirectories in train, val, and test folders
        os.makedirs(os.path.join(output_dir, "train", class_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val", class_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", class_name), exist_ok=True)
        
        # Copy images to their respective folders
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, "train", class_name, img))
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, "val", class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, "test", class_name, img))

print("Dataset split completed.")