import cv2
import numpy as np
import csv
import glob
import os

# Directory containing the images
image_directory = "./test_images"

# List all L_<distance>.png files (assume corresponding R_<distance>.png files exist)
l_files = glob.glob(os.path.join(image_directory, "L_*.png"))

# Prepare to store data
data = []

for l_file in l_files:
    # Extract the ground truth height from the filename
    # height_str = os.path.basename(l_file).split('_')[1].split('.')[0]
    height_str = os.path.basename(l_file).split('_')[1].split('.png')[0]
    ground_truth_height = float(height_str)

    # Initialize the row with the ground truth height
    row = {'Ground Truth Height': ground_truth_height, 'L (pixels)': None, 'R (pixels)': None}

    # Process the L_<distance>.png file
    image_l = cv2.imread(l_file)
    if image_l is not None:
        blue_channel, _, _ = cv2.split(image_l)
        brown_mask = (blue_channel < 200).astype(np.uint8) * 255
        contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, _, w, _ = cv2.boundingRect(contour)
            center_x_l = float(x + w / 2)
            row['L (pixels)'] = f"{center_x_l:.3f}"

    # Find the corresponding R_<distance>.png file
    r_file = l_file.replace('L_', 'R_')
    if os.path.exists(r_file):
        image_r = cv2.imread(r_file)
        if image_r is not None:
            blue_channel, _, _ = cv2.split(image_r)
            brown_mask = (blue_channel < 200).astype(np.uint8) * 255
            contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                x, _, w, _ = cv2.boundingRect(contour)
                center_x_r = float(x + w / 2)
                row['R (pixels)'] = f"{center_x_r:.3f}"

    # Add the row to the data list
    data.append(row)

# Sort data by Ground Truth Height
data.sort(key=lambda x: x['Ground Truth Height'])

# Prepare CSV file for writing
csv_file_path = os.path.join(image_directory, "pixel_distances_sorted.csv")
with open(csv_file_path, mode='w', newline='') as csvfile:
    fieldnames = ['Ground Truth Height', 'L (pixels)', 'R (pixels)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()

    # Write sorted data to CSV
    for row in data:
        writer.writerow(row)

print(f"Sorted pixel distances written to {csv_file_path}")
