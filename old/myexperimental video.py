import cv2
import numpy as np

# Load your image
cap = cv2.VideoCapture("Play 1.mp4")

# Check if the video was opened successfully
if not cap.isOpened():
    print("Could not open the video.")
    exit()

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

# Get the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Only update coordinates, actual frame update happens in main loop
        global mouse_x, mouse_y
        mouse_x, mouse_y = x, y

# Initialize mouse coordinates
mouse_x, mouse_y = 0, 0
cv2.namedWindow("Warped")
cv2.setMouseCallback("Warped", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Warp the perspective
    warped = cv2.warpPerspective(frame, M, (1920, 1080))
    
    # Add coordinates text to frame
    display_img = warped.copy()
    text = f'({mouse_x}, {mouse_y})'
    cv2.putText(display_img, text, (mouse_x+10, mouse_y+10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Warped", display_img)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
