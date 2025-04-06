import os

# Define the combined directory
combined_dir = r'D:\CADT\CapstoneProjectI\ml__model\data\Rice Diseases\white background'

# Initialize a dictionary to hold the counts
class_counts = {}

# Count images in each subfolder
for folder in os.listdir(combined_dir):
    folder_path = os.path.join(combined_dir, folder)
    if os.path.isdir(folder_path):
        # Count the number of image files (assuming .jpg and .png formats)
        image_count = len([file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png'))])
        class_counts[folder] = image_count

# Print the number of images for each class
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")