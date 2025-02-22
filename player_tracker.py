import cv2
import numpy as np
import math

image = cv2.imread('frame.png', cv2.IMREAD_UNCHANGED)
width = image.shape[1]
height = image.shape[0]
width = int(width / 3)
height = int(height / 3)
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
lines = cv2.HoughLinesP(edges, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=50, 
                        minLineLength=100, 
                        maxLineGap=10)
# Create a copy of the original image to draw the detected lines
line_image = image.copy()

if lines is not None:
    avg_angle = 0
    left_most = None
    right_most = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if left_most is None or x1 < left_most[0]:
            left_most = (x1, y1, x2, y2)
        if right_most is None or x2 > right_most[2]:
            right_most = (x1, y1, x2, y2)
        # Optionally, you can filter lines by their angle:
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        # For instance, you can ignore extremely steep or shallow lines if needed:
        # if not (abs(angle) < 10 or abs(angle) > 80):
        #     continue
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2) # Compute the direction vector of the line


# Display the original image, the edges, and the image with detected lines
cv2.imshow("Original Image", image)
cv2.imshow("Edges", edges)
cv2.imshow("Detected Lines", line_image)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
