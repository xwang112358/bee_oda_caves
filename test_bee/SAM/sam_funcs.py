import numpy as np
import cv2 
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 



def get_seed_points(image_path, num_seeds):
    seed_points = []
    point_labels = []
    
    # Define a callback function to handle mouse events
    def on_mouse(event, x, y, flags, param):
        nonlocal seed_points, point_labels
        
        if event == cv2.EVENT_LBUTTONDOWN:  # Left-click event
            seed_points.append([x, y])
            point_labels.append(1)
            cv2.circle(img_side, (x, y), 3, (0, 255, 0), -1)
            
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right-click event
            seed_points.append([x, y])
            point_labels.append(0)
            cv2.circle(img_side, (x, y), 3, (0, 0, 255), -1)

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
    point_labels = np.array(point_labels)
    
    # Destroy the window
    cv2.destroyAllWindows()
    
    return seed_points, point_labels



def crop_box(image_path):
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

    return np.array([x1,y1,x2,y2])