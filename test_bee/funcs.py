import cv2
import numpy as np
from PIL import Image
from pytesseract import pytesseract
import re

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract

def get_seed_points(image_path, num_seeds):
    seed_points = []
    
    # Define a callback function to handle mouse events
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            seed_points.append([x, y])
            cv2.circle(img_side, (x, y), 3, (0, 255, 0), -1)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Image", img_side)
    
    # Load the image
    img = cv2.imread(image_path)
    img_side = img.copy()
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img_side)
    
    # Set the callback function for the window
    cv2.setMouseCallback("Image", on_mouse)
    
    # Wait for the user to select the seed points
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or len(seed_points) >= num_seeds:
            break
    
    # Convert the list of seed points to a numpy array
    seed_points = np.array(seed_points)

    # Destroy the window
    cv2.destroyAllWindows()
    
    return seed_points


def crop_scale(image_path):
    img_side = cv2.imread(image_path)

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

def get_pixel_size_mm(image_path):
    img = Image.open(image_path)
    num = int(pytesseract.image_to_string(img, config= 'digits'))
    print(f"The unit number is {num}")

    scale_img = cv2.imread(image_path)
    scale_gray = cv2.cvtColor(scale_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(scale_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = scale_img.shape[0] * scale_img.shape[1]

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area < 0.95*img_area:
            max_area = area
            largest_contour = contour

    # Find the bounding box of the largest contour
    # w is the width of scale bar
    x,y,w,h = cv2.boundingRect(largest_contour)
    print(f"Width of the rectangle in pixels: {w}")
    print(f"The pixel size is {round(num/w, 6)} pixel per mm")

    return round(num/w, 6)

def get_pixel_size_mm_test(image_path, min_aspect_ratio = 40, extension = 300):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    bg = np.full_like(img, 255)
    fg = cv2.bitwise_and(bg, bg, mask=mask_inv)
    gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)    

    # test
    # cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
    # cv2.imshow("binary", binary_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # width = 1
    # num = 0


    for contour in contours:
        # Find the bounding rectangle of the contour
        x,y,w,h = cv2.boundingRect(contour)
        
        # Calculate the aspect ratio of the bounding rectangle
        rect_aspect_ratio = float(w)/h
        
        # Check if the aspect ratio of the bounding rectangle is larger than the minimum aspect ratio we're looking for
        if rect_aspect_ratio > min_aspect_ratio:
            width = w
            new_height = h + extension
            y_fixed = y + h 
            y_new = y_fixed - new_height
            
            if y_new < 0:
                img_rect = cv2.rectangle(img.copy(), (x, 0), (x+w+5, y_fixed), (0, 255, 0), 5)
            
            else:
                print(y_new)
                img_rect = cv2.rectangle(img.copy(), (x, y_new), (x+w+5, y_fixed), (0, 255, 0), 5)

            # test cropped area
            # cv2.namedWindow("Rectangle", cv2.WINDOW_NORMAL)
            # cv2.imshow('Rectangle', img_rect)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            img_crop = binary_image[0:y_fixed, x:x+w]
            num_str = pytesseract.image_to_string(img_crop, config= 'digits')
    
            # if OCR fails
            if not num_str or not any(char.isdigit() for char in num_str):
                num = input("Please enter a digit number for the unit number:")
                if num.isdigit():
                    num = int(num)
                else:
                    print("Invalid input. Please enter a digit number.")
            else:
                num_str = re.sub(r'\D', '', num_str)
                num = int(num_str)


            contours_scale, _ = cv2.findContours(img_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours_scale:
                x,y,w,h = cv2.boundingRect(contour)
                rect_aspect_ratio = float(w)/h
                if rect_aspect_ratio > min_aspect_ratio:

                    print("the unit is", num)
                    print(f"Width of the rectangle in pixels: {w}")
                    print(f"The pixel size is {round(num/w, 6)}")
    
                    return round(num/w, 6)
