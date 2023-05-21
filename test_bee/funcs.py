import cv2
import numpy as np
from PIL import Image
from pytesseract import pytesseract
import re
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
import os

# set path to tesseract.exe
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



def get_pixel_size_mm_test(image_path, min_aspect_ratio = 40, extension = 300,
                           color = 'b'):
    if color not in ['b', 'w']:
        raise ValueError('Invalid input: scale bar color should be "b" or "w"')
    img = cv2.imread(image_path)

    if color == 'b':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg = np.full_like(img, 255)
        fg = cv2.bitwise_and(bg, bg, mask=mask_inv)
        gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)    

    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)[1]
        mask_inv = cv2.bitwise_not(mask)
        bg = np.full_like(img, 255)
        fg = cv2.bitwise_and(img, bg, mask=mask_inv)
        gray_fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.threshold(gray_fg, 10, 255, cv2.THRESH_BINARY)[1]
        binary_image = cv2.bitwise_not(binary_image)
        

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


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
                img_rect = cv2.rectangle(img.copy(), (x, y_new), (x+w+5, y_fixed), (0, 255, 0), 5)

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

                    print(f"The scale bar length is {num} mm")
                    print(f"Width of the rectangle in pixels: {w}")
                    print(f"The pixel size is {round(num/w, 6)}")
    
                    return round(num/w, 6)


# helper function for contour analysis

def angle_between_3_points(pt1, pt2, pt3):
    """Calculate the angle between three points"""
    
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    pt3 = np.array(pt3)
    
    v1 = pt1 - pt2
    v2 = pt3 - pt2
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle = np.arccos(dot_product / (norm_v1 * norm_v2))
    
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

def calculate_slope(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return (y2 - y1) / (x2 - x1)

def find_closest_point(p1, p2, contour):
    # convert points to numpy array
    p1 = np.array(p1)
    p2 = np.array(p2)

    # calculate the line parameters
    line_direction = p2 - p1
    line_length = np.linalg.norm(line_direction)
    if line_length == 0:
        return None, None  # points coincide, no line to search
    line_direction = line_direction / line_length
    line_normal = np.array([-line_direction[1], line_direction[0]])

    # initialize minimum distance and closest point
    min_distance = np.inf
    closest_point = None
    closest_point_index = None

    # iterate over all contour points
    for i in range(len(contour)):
        point = contour[i]
        # calculate the distance from the point to the line
        pt_to_p1 = point[0] - p1
        pt_to_p2 = point[0] - p2
        projection = np.dot(pt_to_p1, line_direction) * line_direction
        perpendicular = pt_to_p1 - projection
        distance = np.linalg.norm(perpendicular)
        
        dot_product = np.dot(pt_to_p1, pt_to_p2)
        if dot_product > 0:
            continue

        # if the distance is smaller than the current minimum distance, update the closest point
        if distance < min_distance:
            min_distance = distance
            closest_point = point[0]
            closest_point_index = i
            
    return closest_point, closest_point_index


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


def find_point_on_line(points, total_distance, start_point, start_index):
    accumulated_distance = 0
    current_point = start_point
    
    for i in range(start_index+1, len(points)):
        distance = math.sqrt((points[i][0] - current_point[0])**2 + (points[i][1] - current_point[1])**2)
        if accumulated_distance + distance >= total_distance:
            update_index = i
            remaining_distance = total_distance - accumulated_distance
            proportion = remaining_distance / distance
            x = current_point[0] + (points[i][0] - current_point[0]) * proportion
            y = current_point[1] + (points[i][1] - current_point[1]) * proportion
            #print(accumulated_distance)
            return update_index, [x, y]
        else:
            accumulated_distance += distance
            current_point = points[i]
    
    # If we reach the end of the array, return the last point
    return points[-1]


def ls_circle(xx,yy):
   asize = np.size(xx)
   #print('Circle input size is ' + str(asize))
   J=np.zeros((asize,3))
   ABC=np.zeros(asize)
   K=np.zeros(asize)

   for ix in range(0,asize):
      x=xx[ix]
      y=yy[ix]

      J[ix,0]=x*x + y*y
      J[ix,1]=x
      J[ix,2]=y
      K[ix]=1.0

   K=K.transpose()
   JT=J.transpose()
   JTJ = np.dot(JT,J)
   InvJTJ=np.linalg.inv(JTJ)

   ABC= np.dot(InvJTJ, np.dot(JT,K))
   #If A is negative, R will be negative
   A=ABC[0]
   B=ABC[1]
   C=ABC[2]

   xofs=-B/(2*A)
   yofs=-C/(2*A)
   R=np.sqrt(4*A + B*B + C*C)/(2*A)
   if R < 0.0: R = -R
   return (xofs,yofs,R)


def get_image_names(folder_path, keyword):
    image_names = []
    ratio_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.gif')) and keyword.lower() in file_name.lower():
            image_names.append(file_name)

    for img_name in image_names:
        zoom_ratio = img_name.split("_")[1]
        ratio_list.append(int(zoom_ratio[:-1]))

    max_zoom_index = ratio_list.index(max(ratio_list))

    return image_names[max_zoom_index]




def find_x2(arc_length, x1, integrand):
    # Integrate the integrand over the interval from x1 to x2 to get the arc length
    # Initialize x2 to be x1
    x2 = x1  
    arc_length_temp = 0.0
    
    while arc_length_temp < arc_length:
        x2 += 1 
        arc_length_temp, _ = quad(integrand, x1, x2)
        
    while arc_length_temp > arc_length:
        x2 -= 0.005
        arc_length_temp, _ = quad(integrand, x1, x2)
            
    return x2