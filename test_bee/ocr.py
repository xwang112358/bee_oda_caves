from PIL import Image
from pytesseract import pytesseract
import cv2
import numpy as np
import pytesseract


# to do: combine ocr.py and test.py to some functions

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract

## crop the scale bar for measurement 
# this step will be replaced by an object detection algorithm
img_side = cv2.imread("UCSB-IZC00039429_3x_hal_lg.jpg")

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", img_side)

# Initialize an empty list to store the seed points
seed_points = []

# Define a callback function to handle mouse events
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        seed_points.append([x, y])
        cv2.circle(img_side, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Image", img_side)

# Set the callback function for the window
cv2.setMouseCallback("Image", on_mouse)

# Wait for the user to select the seed points
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or len(seed_points) >= 2:
        break

# Convert the list of seed points to a numpy array
seed_points = np.array(seed_points)

# Destroy the window
cv2.destroyAllWindows()

x1, y1 = seed_points[0]
x2, y2 = seed_points[1]

cropped_img = img_side[y1:y2, x1:x2]

#--------------------------------------------
# to do: assign name to the image 
#--------------------------------------------

gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

# Create a binary mask of the image
_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# Invert the binary mask
mask_inv = cv2.bitwise_not(mask)

# Create a white background image
bg = np.full_like(cropped_img, 255)

# Copy the background image using the inverted mask
fg = cv2.bitwise_and(bg, bg, mask=mask_inv)

gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)

# Apply thresholding to convert the grayscale image to a binary image
_, binary_cropped_image = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)


cv2.imwrite("./scale_bar/cropped_scale.jpg", binary_cropped_image)

