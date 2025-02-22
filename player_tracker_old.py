import cv2
import numpy as np

# Load and preprocess the image
image = cv2.imread('frame.png', cv2.IMREAD_UNCHANGED)
width = int(image.shape[1] / 3)
height = int(image.shape[0] / 3)
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Convert to grayscale and apply edge detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

# Morphological operation to isolate horizontal hash marks
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))  # Adjust kernel size for horizontal marks
morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Detect contours for hash marks
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter horizontal hash mark contours
hash_marks = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    if 5 < aspect_ratio < 15 and w > 10:  # Adjust aspect ratio and size thresholds as needed
        hash_marks.append((x, y, w, h))

# Create a copy of the image for visualization
output_image = image.copy()

# Draw detected lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Find intersections between lines and hash marks
for (x, y, w, h) in hash_marks:
    hash_center = (x + w // 2, y + h // 2)
    # Draw the hash mark for visualization
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Check if the hash mark center lies on the line segment
        if min(x1, x2) <= hash_center[0] <= max(x1, x2):
            # Calculate the line equation: y = mx + b
            if x2 - x1 != 0:  # Avoid division by zero
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                y_on_line = int(m * hash_center[0] + b)
                if abs(y_on_line - hash_center[1]) < 5:  # Tolerance for intersection
                    # Mark the intersection point
                    cv2.circle(output_image, hash_center, 5, (255, 0, 0), -1)

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Edges", edges)
cv2.imshow("Morphed", morphed)
cv2.imshow("Hash Marks and Lines", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
