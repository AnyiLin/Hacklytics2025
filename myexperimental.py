# start with the video frame loading
# cap = cv2.VideoCapture('football_game.mp4')  # replace with your video path

# # check if video opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video file")
#     exit()

# get the bounding lines of the field (the 5 yard incremental ones)


# get the bounding lines under and over the field (the thick sideline lines)

# rotate the perspective to actually show an overhead view

# use the contours to create a timelapse of the contours overlaid every frame

import cv2
import numpy as np

# Load your image
cap = cv2.VideoCapture("Play 1.mp4")

# Check if the video was opened successfully
if not cap.isOpened():
    print("Could not open the video.")
    exit()

# Read the first frame
ret, image = cap.read()
# image = cv2.imread("frame.png")

# 1. Identify four reference points in the *original* image
#    For example, pick corners or distinct yard-line intersections.
#    Replace (x1,y1), (x2,y2), etc. with your real pixel coordinates.

src_pts = np.float32([
    [175, 255],  # top-left reference
    [1410, 179],  # top-right reference
    [1727, 795],  # bottom-right reference
    [243, 910]   # bottom-left reference
])

dst_pts = np.float32([
    [0, 0],  # top-left reference
    [1920, 0],  # top-right reference
    [1920, 1080],  # bottom-right reference
    [0, 1080]   # bottom-left reference
])

# 3. Get the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 4. Warp the perspective to produce a top-down (rectified) image
warped = cv2.warpPerspective(image, M, (1920, 1080))

output = warped

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Create a copy of the current frame
        display_img = output.copy()
        # Add text showing coordinates
        text = f'({x}, {y})'
        cv2.putText(display_img, text, (x+10, y+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Warped", display_img)

# Replace the final display section with:
cv2.imshow("Warped", output)
cv2.setMouseCallback("Warped", mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()
