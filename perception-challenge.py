import cv2
import numpy as np;
from matplotlib import pyplot as plt

img = cv2.imread("perception-challenge.png") 
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, w, channels = hsv_img.shape
half = w//2

# Divide the image into left and right halves
left_half = hsv_img[:, :half]
# Isolate cones by bright red color
left_mask = cv2.inRange(left_half,(176, 220, 90), (180, 255, 255)) 
left_contours, _ = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw dots at the centroids of isolated regions on the left side of the image
left_dot_coordinates = []
for contour in left_contours:
    # Calculate the centroid of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        left_dot_coordinates.append((cx, cy))
        
        # Draw a dot at the centroid
        cv2.circle(img, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)

# Use numPy polyfit function to generate line of best fit to dots on left side of image
left_array = np.array(left_dot_coordinates)
m, b = np.polyfit(left_array[:, 0], left_array[:, 1], 1)
x1 = 0
y1 = int(m * x1 + b)
x2 = img.shape[1]
y2 = int(m * x2 + b)
cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Repeat: Draw dots at the centroids of the highlighted regions on the right side of the image
right_half = hsv_img[:, half:]
# Isolate cones by bright red color
right_mask = cv2.inRange(right_half,(176, 220, 90), (180, 255, 255))
right_contours, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw dots at the centroids of isolated regions on the right side of the image
right_dot_coordinates = []
for contour in right_contours:
    # Calculate the centroid of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        right_dot_coordinates.append((cx, cy))
        
        # Draw a dot at the centroid 
        cv2.circle(img, (cx + w//2, cy), radius=5, color=(0, 255, 0), thickness=-1)

right_array = np.array(right_dot_coordinates)
m, b = np.polyfit(right_array[:, 0], right_array[:, 1], 1)
x1 = 0
y1 = int(m * x1 + b)
x2 = img.shape[1]
y2 = int(m * x2 + b)
cv2.line(img, (x1 + w//2, y1), (x2 + w//2, y2), (0, 0, 255), 2)

# Show results
color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
plt.imshow(color_img)
plt.show()