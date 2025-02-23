import cv2
import numpy as np

def mask_players(frame, player_detector=None):
    """
    Optional: Use a pretrained player detector to mask out players.
    For demonstration purposes, this function returns a mask that is all 1’s.
    In a real application, you might use a deep learning detector here.
    """
    # For now, we return a mask that doesn't mask anything.
    return np.ones(frame.shape[:2], dtype=np.uint8)

def extract_field_lines(frame, player_detector=None):
    # (1) Optionally mask out players:
    player_mask = mask_players(frame, player_detector)
    
    # (2) Convert to HSV color space to segment white.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define threshold for white.
    # These values can be tuned based on lighting conditions.
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # (3) Apply the player mask to ignore regions where players are detected.
    # Assume player_mask has 0 where players are, 1 (or 255) elsewhere.
    # Here we assume player_mask is binary (0 or 1), so we scale it to 255.
    white_mask = cv2.bitwise_and(white_mask, white_mask, mask=player_mask)
    
    # (4) Use morphological operations to close gaps.
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    # (5) Edge Detection: use Canny.
    edges = cv2.Canny(closed_mask, 50, 150, apertureSize=3)
    
    # (6) Detect line segments with Hough Transform.
    lines = cv2.HoughLinesP(edges, 
                            rho=1, 
                            theta=np.pi/180, 
                            threshold=80, 
                            minLineLength=50, 
                            maxLineGap=20)
    return closed_mask, edges, lines

def draw_lines(frame, lines):
    """
    Draw detected lines on a copy of the frame.
    """
    line_img = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    return line_img

def main():
    cap = cv2.VideoCapture('Play 1.mp4')
    
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # (Extract the field lines)
        mask, edges, lines = extract_field_lines(frame)
        
        # (Optionally display the results)
        line_img = draw_lines(frame, lines)
        cv2.imshow('Original Frame', frame)
        cv2.imshow('White Mask', mask)
        cv2.imshow('Edges', edges)
        cv2.imshow('Detected Lines', line_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
