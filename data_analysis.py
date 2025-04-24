import os
import cv2

# 
# data > yolo > images 에 대한 분석

# Directories containing the images for train and val
image_dirs = ["data/yolo/augmentation/images", "data/yolo/augmentation/labels"]

# Dictionary to store resolutions and their counts
resolution_counts = {}

# Iterate through all directories
for image_dir in image_dirs:
    # Iterate through all files in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            # Read the image
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            
            # Get the resolution (width, height)
            resolution = (image.shape[1], image.shape[0])
            
            # Update the resolution count
            if resolution in resolution_counts:
                resolution_counts[resolution] += 1
            else:
                resolution_counts[resolution] = 1

# Print the resolutions and their counts
for resolution, count in resolution_counts.items():
    print(f"Resolution {resolution}: {count} images")
