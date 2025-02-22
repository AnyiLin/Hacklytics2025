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
    for line in lines:
        print(line)
        x1, y1, x2, y2 = line[0]
        # Optionally, you can filter lines by their angle:
        # angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        # For instance, you can ignore extremely steep or shallow lines if needed:
        # if not (abs(angle) < 10 or abs(angle) > 80):
        #     continue
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2) # Compute the direction vector of the line
        dx = x2 - x1
        dy = y2 - y1
        
        # Get a perpendicular vector: (-dy, dx) is perpendicular to (dx, dy)
        perp_vector = np.array([-dy, dx])
        
        # Normalize the perpendicular vector to get a unit vector (optional)
        if np.linalg.norm(perp_vector) != 0:
            perp_unit = perp_vector / np.linalg.norm(perp_vector)
        else:
            perp_unit = perp_vector  # Return the zero vector if points are identical.
        perp_unit *= 100
        p_x1a = int(x1 - perp_unit[0])
        p_y1a = int(y1 - perp_unit[1])
        p_x1b = int(x1 + perp_unit[0])
        p_y1b = int(y1 + perp_unit[1])
        p_x2a = int(x2 - perp_unit[0])
        p_y2a = int(y2 - perp_unit[1])
        p_x2b = int(x2 + perp_unit[0])
        p_y2b = int(y2 + perp_unit[1])
        cv2.line(line_image, (p_x1a, p_y1a), (p_x1a, p_y1a), (0, 255, 0), 2)
        cv2.line(line_image, (p_x2b, p_y2b), (p_x2b, p_y2b), (0, 255, 0), 2)
cv2.line(line_image, (0,0), (width, height), (0, 255, 0), 2)


# Display the original image, the edges, and the image with detected lines
cv2.imshow("Original Image", image)
cv2.imshow("Edges", edges)
cv2.imshow("Detected Lines", line_image)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
