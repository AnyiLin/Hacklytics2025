import sys
import os
import gc
from ultralytics import YOLO
import torch
from PIL import Image
import cv2
import numpy as np

def preprocess_frame(frame):
    # Convert to LAB color space for better light handling
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel for better contrast in shadows and highlights
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels back
    lab = cv2.merge((l,a,b))
    balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply slight Gaussian blur to reduce glare
    blurred = cv2.GaussianBlur(balanced, (3,3), 0)
    
    # Adjust gamma for better white jersey detection
    gamma = 0.85  # Reduce intensity of bright areas
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(blurred, table)
    
    # Increase saturation slightly to help differentiate players from field
    hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2)  # Increase saturation by 20%
    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return final

# Function to identify players using YOLOv10
def identify_players_yolo(image, model, device):
    # Preprocess the image
    processed_image = preprocess_frame(image)
    
    with torch.no_grad():
        results = model(processed_image)  # YOLOv10 inference
        people = []
        
        for result in results:
            boxes = result.boxes.data  # returns boxes object for bbox outputs
            for box in boxes:
                if box[5] == 0:  # class 0 is person
                    x1, y1, x2, y2, conf, cls = box
                    people.append([int(x1), int(y1), int(x2), int(y2)])
        
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return people

# Load YOLOv10 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO('yolov10m.pt')  # or yolov10l.pt or yolov10x.pt
yolo_model.to(device)

# Process frames in batches
BATCH_SIZE = 4
frames_buffer = []
processed_frames = []

cap = cv2.VideoCapture('Play 1.mp4')

while True:
    # Read frames into buffer
    while len(frames_buffer) < BATCH_SIZE:
        ret, frame = cap.read()
        if not ret:
            break
        frames_buffer.append(frame)
    
    if not frames_buffer:
        break
    
    # Process batch
    for frame in frames_buffer:
        # Process and detect
        yolo_people = identify_players_yolo(frame, yolo_model, device)
        
        # For debugging: show preprocessed frame
        processed = preprocess_frame(frame)
        cv2.imshow("Preprocessed Frame", cv2.resize(processed, 
                   (int(processed.shape[1]/2), int(processed.shape[0]/2))))
        
        # Draw boxes
        for box in yolo_people:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
        
        # Resize for display
        width = int(frame.shape[1] / 2)
        height = int(frame.shape[0] / 2)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        processed_frames.append(frame)
    
    # Display processed frames
    for frame in processed_frames:
        cv2.imshow("Image with Bounding Boxes", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
    
    # Clear buffers
    frames_buffer.clear()
    processed_frames.clear()
    
    # Force garbage collection
    gc.collect()

cap.release()
cv2.destroyAllWindows()
