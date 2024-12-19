import os
import cv2
import numpy as np
import pandas as pd
from funcs import (
    get_seed_points,
    get_pixel_size_mm_test,
    get_image_names
)
from analysis_tools import Eye
import csv

bee_df = pd.read_csv("results/raw_apidae.csv")
bee_catalog_num = input("Enter the Catalog Number: \n")
img_folder_path = "data/" + bee_catalog_num

# retrieve bee information
family = bee_df.query('catalogNumber == @bee_catalog_num')['Family'].item()
scientific_name = bee_df.query('catalogNumber == @bee_catalog_num')['scientificName'].item()
sex = bee_df.query('catalogNumber == @bee_catalog_num')['sex'].item()
print("Bee Catalog Number:", bee_catalog_num)
print("Family:", family)
print("Scientific Name:", scientific_name) 
print("Sex:", sex)

# get image path from the folder
image_view = input("Loading 'hef' or 'hal' image for sampling ommatidia diameter: ")

image_names = get_image_names(img_folder_path, image_view)
img_path = img_folder_path + "/" + image_names

# split the information for the image name, e.g. UCSB-IZC00035539_2x_hal_lg.jpg
org = image_names.split("-")[0]
bee_id = image_names.split("-")[1].split("_")[0]
zoom_ratio = image_names.split("_")[1]

# get the pixel size via tesseract
pixel_size = get_pixel_size_mm_test(img_path, color='b')
#print(pixel_size)

if pixel_size == None:
    pixel_size = float(input("enter the pixel size: \n"))

# manually create a polygon mask to measure ommatidia distances in that area
image = cv2.imread(img_path)
points = get_seed_points(img_path, 10)
mask = np.zeros_like(image[:,:,0])
cv2.fillPoly(mask, [points], (255, 255, 255))
cv2.imwrite("./created_masks/poly_mask.jpg", mask)

eye = Eye(img_path, mask_fn="./created_masks/poly_mask.jpg", 
          pixel_size=pixel_size)

eye.load_memmap()
eye.get_eye_outline()
eye.get_eye_dimensions(display=False)
eye.crop_eye(use_ellipse_fit=False)
eye.crop_eye(use_ellipse_fit=True)
# use the cropped image
cropped_eye = eye.eye

# run the ommatidia_detecting_algorithm
# check the hyper-parameters from original repo: https://github.com/jpcurrea/ODA

# check f'{img_path.split(".")[0]}_oda.svg' not exists
sample_count = 0
while os.path.exists(f'{img_path.split(".")[0]}_oda{sample_count}.svg'):
    sample_count += 1

# Create single ODA file with the next available number
cropped_eye.oda(bright_peak=False, high_pass=True, plot=True,
            plot_fn=f'{img_path.split(".")[0]}_oda{sample_count}.svg', manual_edit=True)

measurement_result = cropped_eye.get_result()
area = measurement_result[0]
count = measurement_result[1]
sample_diameter = measurement_result[2]
sample_std = measurement_result[3]
fft_diameter = measurement_result[4]

# save the result to a csv file
result_gate = input("Do you want to save the result? (y/n) \n")
if result_gate == 'y':
    # Create new row with the measurement data
    new_row = bee_df[bee_df['catalogNumber'] == bee_catalog_num].iloc[0].copy()
    new_row['sample_diameter_mm'] = sample_diameter
    new_row['sample_std_mm'] = sample_std
    new_row['num_ommatida_sample'] = count
    
    # Convert the Series to a DataFrame with a single row
    new_row_df = pd.DataFrame([new_row])
    
    # If the results file exists, append to it; otherwise create new
    if os.path.exists('results/apidae_results.csv'):
        new_row_df.to_csv('results/apidae_results.csv', mode='a', header=False, index=False)
    else:
        new_row_df.to_csv('results/apidae_results.csv', index=False)
    