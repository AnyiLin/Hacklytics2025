import sys
import gc
import cv2
from yolo10_detection import player_detection


cap = cv2.VideoCapture('Play 1.mp4')
yolo = player_detection()
positions = []

counter = 0

while True:
    # Read frames into buffer
    ret, frame = cap.read()
    if not ret:
        break

    new_positions = yolo.find_players(frame)

    for i in new_positions:
        positions.append((i[0], i[1], counter))

    for i in positions:
        cv2.circle(frame, (int(i[0]), int(i[1])), 5, (0 + i[2] * 2, 0 + i[2] * 2, 255), -1)
    
    counter += 1
    # Resize for display
    width = int(frame.shape[1] / 2)
    height = int(frame.shape[0] / 2)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow("Image with Bounding Boxes", frame)
    cv2.waitKey(0)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()

    # Force garbage collection
    gc.collect()

cap.release()
cv2.destroyAllWindows()
