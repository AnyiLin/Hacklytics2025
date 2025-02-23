import cv2
import numpy as np
from line_detection_copy import find_lines

# Load your image
cap = cv2.VideoCapture("Play 1.mp4")

# Check if the video was opened successfully
if not cap.isOpened():
    print("Could not open the video.")
    exit()

# src_pts = np.float32([
#     [175, 255],  # top-left reference
#     [1410, 179],  # top-right reference
#     [1727, 795],  # bottom-right reference
#     [243, 910]   # bottom-left reference
# ])

# dst_pts = np.float32([
#     [0, 0],  # top-left reference
#     [1920, 0],  # top-right reference
#     [1920, 1080],  # bottom-right reference
#     [0, 1080]   # bottom-left reference
# ])

# # Get the perspective transformation matrix
# M = cv2.getPerspectiveTransform(src_pts, dst_pts)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    
    # Resize frame if it's too large
    if frame.shape[1] > 1920:  # if width is greater than 1920
        scale = 1920 / frame.shape[1]
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    try:
        # Warp the perspective
        # warped = cv2.warpPerspective(frame, M, (1920, 1080))
        
        lines, angles = find_lines(frame)
        
        # Display the warped image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line  # Extract points from the nested array
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
        # cv2.imshow("Warped", warped)
        cv2.imshow("frame", frame)
        
        # Break loop on 'q' press
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        continue

cap.release()
cv2.destroyAllWindows()
