# import sys
# sys.path.append("../src/")

#from ODA import *
import cv2
from funcs import *
import csv
from analysis_tools import *


img_info = input("Enter the image path: \n")
img_path = "raw_images/" + img_info


# split the information
org = img_info.split("-")[0]
bee_id = img_info.split("-")[1].split("_")[0]
zoom_ratio = img_info.split("_")[1]


pixel_size = get_pixel_size_mm_test(img_path)
print(pixel_size)

# manually create a polygon mask to measure ommatidia distances in that area
image = cv2.imread(img_path)
points = get_seed_points(img_path, 6)
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

file_exists = os.path.isfile('results/output.csv')
headers = ["Org", "Bee ID", "Zoom Ratio", "Pixel Size",
           "eye_area", "N", "sample_diameter", "sample_std",
           "fft_diameter"]
# Define the data rows
rows = [
    [org, bee_id, zoom_ratio, pixel_size, area, count, sample_diameter, sample_std, fft_diameter],
    # Add more rows as needed
]
# Open the CSV file for appending
with open('results/output.csv', mode='a', newline='') as file:
    # Create CSV writer object
    writer = csv.writer(file)

    # Write the column headers to the CSV file
    if not file_exists:
        writer.writerow(headers)

    # Write the data rows to the CSV file
    for row in rows:
        writer.writerow(row)