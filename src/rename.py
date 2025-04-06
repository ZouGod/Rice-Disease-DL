import os

# Define the base directory containing all disease folders
base_dir = r"D:\CADT\CapstoneProjectI\ml__model\data\raw_images"

# Get a list of all subfolders (disease folders) in the base directory
disease_folders = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]

# Loop through each disease folder
for disease in disease_folders:
    disease_dir = os.path.join(base_dir, disease)
    
    # Convert the disease name to lowercase
    disease_lower = disease.lower()
    
    # List all files in the disease folder
    images = os.listdir(disease_dir)
    
    # Rename files with the naming convention "disease_lower_imageID.ext"
    for i, image_name in enumerate(images):
        # Check if the file is an image (optional)
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Define the new name
            new_name = f"{disease_lower}_{i+1:04d}{os.path.splitext(image_name)[1]}"
            # Rename the file
            os.rename(os.path.join(disease_dir, image_name), os.path.join(disease_dir, new_name))
            print(f"Renamed {image_name} to {new_name} in {disease_dir}")
    print(f"Completed renaming for {disease_dir}")

print("Renaming completed for all folders.")