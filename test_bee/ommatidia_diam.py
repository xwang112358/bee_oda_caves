# import sys
# sys.path.append("../src/")

#from ODA import *
import cv2
from funcs import *
import csv
from analysis_tools import *

bee_df = pd.read_csv("results/apidae.csv")
bee_catalog_num = input("Enter the Catalog Number: \n")
img_folder_path = "data/" + bee_catalog_num


family = bee_df.query('catalogNumber == @bee_catalog_num')['Family'].item()
scientific_name = bee_df.query('catalogNumber == @bee_catalog_num')['scientificName'].item()
sex = bee_df.query('catalogNumber == @bee_catalog_num')['sex'].item()
print("Bee Catalog Number:", bee_catalog_num)
print("Family:", family)
print("Scientific Name:", scientific_name) 
print("Sex:", sex)

# get image path from the folder
keyword = input("Study 'hef' or 'hal' image: ")

image_names = get_image_names(img_folder_path, keyword)

img_path = img_folder_path + "/" + image_names


# split the information
org = image_names.split("-")[0]
bee_id = image_names.split("-")[1].split("_")[0]
zoom_ratio = image_names.split("_")[1]

# get the pixel size
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
cropped_eye.oda(bright_peak=False, high_pass=True, plot=True,
                plot_fn='./IZC00046621.svg', manual_edit=True)


measurement_result = cropped_eye.get_result()
area = measurement_result[0]
count = measurement_result[1]
sample_diameter = measurement_result[2]
sample_std = measurement_result[3]
fft_diameter = measurement_result[4]


# save the result to a csv file
result_gate = input("Do you want to save the result? (y/n) \n")
if result_gate == 'y':
    row_index = bee_df[bee_df['catalogNumber'] == bee_catalog_num].index.item()
    bee_df.loc[row_index, 'sample_diameter_mm'] = sample_diameter
    bee_df.loc[row_index, 'sample_std_mm'] = sample_std
    bee_df.loc[row_index, 'num_ommatida_sample'] = count

    bee_df.to_csv('results/apidae.csv', index=False)