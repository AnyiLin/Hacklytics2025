import cv2
import numpy as np
def angle_between(p1, p2, p3):
    """
    Computes the angle (in degrees) at point p2 given three 2D points.
    p1, p2, p3 are arrays/lists/tuples representing (x, y) coordinates.
    """
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # Clamp the cosine to the valid range to avoid numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def is_rectangular(contour, angle_tolerance=15):
    """
    Returns True if the contour approximates to a rectangle.
    angle_tolerance sets how far the angles may deviate from 90°.
    """
    # Approximate the contour
    peri = cv2.arcLength(contour, True)
    epsilon = 0.02 * peri  # You may need to tune this multiplier.
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if approximated polygon has 4 vertices and is convex.
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return False, approx

    # Check angles at each vertex
    angles = []
    for i in range(4):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % 4][0]
        p3 = approx[(i + 2) % 4][0]
        angle = angle_between(p1, p2, p3)
        angles.append(angle)
    
    # For a rectangle, all angles should be close to 90 degrees.
    for angle in angles:
        if abs(angle - 90) > angle_tolerance:
            return False, approx

    return True, approx
def is_white_or_green(roi):
    """
    Determines if the region of interest (roi) is predominantly white or green.
    
    White condition: high brightness (value) and low saturation.
    Green condition: Hue within a green range (approx. 35 to 85) and sufficient saturation.
    """
    # Convert ROI to HSV color space.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v, _ = cv2.mean(hsv)
    print(mean_h)
    print(mean_s)
    print(mean_v)
    
    # White: bright and low saturation.
    if mean_v > 70 and mean_s < 30:
        return True
    
    # Green: hue between 35 and 85 and with a decent saturation.
    if 35 <= mean_h <= 85 and mean_s > 70:
        return True

    return False


# asdf = cv2.imread("green.png", cv2.IMREAD_UNCHANGED)
# print(is_white_or_green(asdf))

# Open the MP4 file
cap = cv2.VideoCapture('Play 1.mp4')

# Create a background subtractor (MOG2 is often a good choice)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:  # Break the loop if there are no frames left
        break

    # Apply the background subtractor to get the foreground mask
    fgmask = fgbg.apply(frame)
    
    # Apply morphological operations to reduce noise in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours from the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Filter out small contours that may correspond to noise
        if  300 < cv2.contourArea(cnt) and cv2.contourArea(cnt) < 1000:
            # Compute the bounding rectangle from the approximated polygon.
            x, y, w, h = cv2.boundingRect(cnt)
            # Extract the ROI from the original frame.
            roi = frame[y:y+h, x:x+w]
            if not is_white_or_green(roi):
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame with detected objects
    cv2.imshow('Tracked Objects', frame)
    cv2.imshow('Foreground Mask', fgmask)
    
    # Exit if the Escape key is pressed
    if cv2.waitKey(0) & 0xFF == 27:
        break
    cv2.waitKey(0)

# Clean up
cap.release()
cv2.destroyAllWindows()

