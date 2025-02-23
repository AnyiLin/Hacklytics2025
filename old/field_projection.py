import cv2
import numpy as np
import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

def detect_field_numbers(frame, reader):
    """
    Preprocess the input frame and use pytesseract to detect numbers on the field.
    Returns a dictionary mapping detected number to its center (x, y) in image coordinates,
    and an annotated copy of the frame for visualization.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise with a small Gaussian blur
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Use adaptive thresholding to make the numbers stand out
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Convert the image from BGR (OpenCV default) to RGB
    image_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    # Read text from the image
    results = reader.readtext(image_rgb)
    # Print the results
    for bbox, text, confidence in results:
        print(f"Detected text: {text}")
        print(f"Bounding box: {bbox}")
        print(f"Confidence: {confidence}\n")

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
        warped = cv2.warpPerspective(frame, M, (1920, 1080))
        
        # Detect numbers and get the annotated image
        detect_field_numbers(warped, reader)
        
        # Display the warped image
        cv2.imshow("Warped", warped)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        continue

cap.release()
cv2.destroyAllWindows()
