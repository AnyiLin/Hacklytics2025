import cv2
import sys
import numpy as np
from line_detection import find_lines
from yolo10_detection import player_detection

# Load your image
cap = cv2.VideoCapture("Play 1.mp4")
yolo = player_detection()
positions = []
counter = 0
on_first_frame = True
first_frame = None

# Check if the video was opened successfully
if not cap.isOpened():
    print("Could not open the video.")
    exit()

while True:
    ret, frame = cap.read()
    if on_first_frame:
        first_frame = frame
    if not ret:
        break
    

    lines, angles = find_lines(frame)
    new_positions = yolo.find_players(frame)
    
    if on_first_frame:
        on_first_frame = False

    
    # Display the warped image
    for line in lines:
        x1, y1, x2, y2 = line  # Extract points from the nested array
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    for i in new_positions:
        positions.append((i[0], i[1], counter))
    for i in positions:
        cv2.circle(frame, (int(i[0]), int(i[1])), 5, (0 + i[2] * 2, 0 + i[2] * 2, 255), -1)
    counter += 1
    # cv2.imshow("Warped", warped)
    cv2.imshow("frame", frame)
    
    cv2.waitKey(0)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
cap.release()
cv2.destroyAllWindows()
