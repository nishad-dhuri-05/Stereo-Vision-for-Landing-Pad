import cv2
import numpy as np
import csv
import glob
import os

# Directory containing the images
image_directory = "../images_hidden/images"

# Initialize the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()

# Fine-tuning detection parameters
aruco_params.adaptiveThreshConstant = 7  # Adjusting threshold constant
aruco_params.adaptiveThreshWinSizeMin = 5
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10

# Function to refine corners to subpixel accuracy
def refine_subpixel(image, corner):
    # Convert the image to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the search window size and termination criteria
    win_size = (5, 5)
    zero_zone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 0.001)

    # Refine the corner to subpixel accuracy
    corner = np.float32([corner])
    refined_corner = cv2.cornerSubPix(image, corner, win_size, zero_zone, criteria)
    return refined_corner[0]

# Preprocess image to improve detection, including edge sharpening
def preprocess_image(image):
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Optionally increase contrast
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=20)  # Increase contrast and brightness
    
    # Sharpen the image (using a simple sharpening kernel)
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    image = cv2.filter2D(image, -1, sharpening_kernel)  # Apply sharpening
    #rotate image by 180 degrees in anti-clockwise direction
    image = cv2.rotate(image, cv2.ROTATE_180)
    return image

# List all L_<distance>.png files (assume corresponding R_<distance>.png files exist)
l_files = glob.glob(os.path.join(image_directory, "L_*.tiff"))

# Prepare to store data
data = []

for l_file in l_files:
    # Extract the ground truth height from the filename
    height_str = os.path.basename(l_file).split('_')[1].split('.tiff')[0]
    ground_truth_height = float(height_str)

    # Initialize the row with the ground truth height
    row = {'Ground Truth Height': ground_truth_height, 'L (pixels)': None, 'R (pixels)': None}

    # Process the L_<distance>.png file
    image_l = cv2.imread(l_file, cv2.IMREAD_GRAYSCALE)
    if image_l is not None:
        # Preprocess image to reduce noise, increase contrast, and sharpen edges
        image_l = preprocess_image(image_l)

        # Detect ArUco markers in the image
        corners_l, ids_l, _ = cv2.aruco.detectMarkers(image_l, aruco_dict, parameters=aruco_params)
        
        # Draw detected markers for visual debugging
        if ids_l is not None and len(corners_l) > 0:
            cv2.aruco.drawDetectedMarkers(image_l, corners_l, ids_l)
            cv2.imshow(f"Detected ArUco markers in L file {l_file}", image_l)
            cv2.waitKey(1000)  # Pause to view the results, adjust the value if needed

        if ids_l is not None and len(corners_l) > 0:
            # Get the first detected ArUco marker's center
            corner_l = corners_l[0][0]
            center_x_l = np.mean(corner_l[:, 0])
            center_y_l = np.mean(corner_l[:, 1])

            # Refine the detected center to subpixel accuracy
            refined_center_l = refine_subpixel(image_l, (center_x_l, center_y_l))

            row['L (pixels)'] = f"{refined_center_l[0]:.3f}"

            # Print detected values for debugging
            print(f"Detected in L file {l_file}: Center (x, y) = ({center_x_l:.2f}, {center_y_l:.2f})")
            print(f"Refined center in L file {l_file}: ({refined_center_l[0]:.2f}, {refined_center_l[1]:.2f})")
        else:
            print(f"No ArUco markers detected in L file {l_file}")

    # Find the corresponding R_<distance>.png file
    r_file = l_file.replace('L_', 'R_')
    if os.path.exists(r_file):
        image_r = cv2.imread(r_file, cv2.IMREAD_GRAYSCALE)
        if image_r is not None:
            # Preprocess image to reduce noise, increase contrast, and sharpen edges
            image_r = preprocess_image(image_r)

            # Detect ArUco markers in the image
            corners_r, ids_r, _ = cv2.aruco.detectMarkers(image_r, aruco_dict, parameters=aruco_params)
            
            # Draw detected markers for visual debugging
            if ids_r is not None and len(corners_r) > 0:
                cv2.aruco.drawDetectedMarkers(image_r, corners_r, ids_r)
                cv2.imshow(f"Detected ArUco markers in R file {r_file}", image_r)
                cv2.waitKey(100)

            if ids_r is not None and len(corners_r) > 0:
                # Get the first detected ArUco marker's center
                corner_r = corners_r[0][0]
                center_x_r = np.mean(corner_r[:, 0])
                center_y_r = np.mean(corner_r[:, 1])

                # Refine the detected center to subpixel accuracy
                refined_center_r = refine_subpixel(image_r, (center_x_r, center_y_r))

                row['R (pixels)'] = f"{refined_center_r[0]:.3f}"

                # Print detected values for debugging
                print(f"Detected in R file {r_file}: Center (x, y) = ({center_x_r:.2f}, {center_y_r:.2f})")
                print(f"Refined center in R file {r_file}: ({refined_center_r[0]:.2f}, {refined_center_r[1]:.2f})")
            else:
                print(f"No ArUco markers detected in R file {r_file}")
    
    # Add the row to the data list
    data.append(row)

# Sort data by Ground Truth Height
data.sort(key=lambda x: x['Ground Truth Height'])

# Prepare CSV file for writing
csv_file_path = os.path.join(image_directory, "aruco_subpixel.csv")
with open(csv_file_path, mode='w', newline='') as csvfile:
    fieldnames = ['Ground Truth Height', 'L (pixels)', 'R (pixels)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()

    # Write sorted data to CSV
    for row in data:
        writer.writerow(row)

print(f"Sorted pixel distances with ArUco subpixel accuracy written to {csv_file_path}")
