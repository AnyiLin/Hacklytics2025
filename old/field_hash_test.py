import cv2
import numpy as np

# Load and convert image
image = cv2.imread("field-hash.png")
# image = cv2.imread("frame.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# More aggressive edge detection specifically for lines
edges = cv2.Canny (
    enhanced, 
    threshold1=150,  # Higher lower threshold
    threshold2=100,  # Higher upper threshold
    apertureSize=3
)

# Dilate edges to connect nearby edges
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

data = []
finalContours = []

for contour in contours:
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if (area > 14 and area < 2000):
        x, y, w, h =  cv2.boundingRect(approx)
        print (x,y,w,h, " > ", abs(w - h) , area)
        if (abs(w - h) < 20):
            finalContours.append(contour)
            data.append(area)

print(data)
    

# Process each contour
median_value = np.median(data)

# Compute the 40th and 60th percentiles.
# np.percentile takes the percentile value as a number from 0 to 100.
percentile_40 = np.percentile(data, 40)
percentile_80 = np.percentile(data, 70)
average = np.average(data)

# Print the results
print(f"Median: {median_value}")
print(f"40th Percentile: {percentile_40}")
print(f"80th Percentile: {percentile_80}")
print(average)



for contour in finalContours:
    # Get perimeter and approximate the contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    # Filter for rectangular shapes (4 points) and minimum area
    area = cv2.contourArea(contour)
    if len(approx) >= 4 and len(approx) <= 8 and area > percentile_40 and area < 2000:
        # Draw the rectangle
        x, y, w, h = cv2.boundingRect(approx)
        # if (x - y)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Optional: Draw the actual contour in a different color
        cv2.drawContours(image, [approx], -1, (0, 0, 255), 2)



# Show results
cv2.imshow("Detected Rectangles", image)
cv2.imshow("Edges", dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()
