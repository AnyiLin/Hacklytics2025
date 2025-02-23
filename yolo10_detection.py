from ultralytics import YOLO
import torch
import cv2
import numpy as np

class player_detection:

    def __init__(self):
        # Load YOLOv10 model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model = YOLO('yolov10m.pt')  # or yolov10l.pt or yolov10x.pt
        self.yolo_model.to(self.device)

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
    def find_players(self, image):
        # Preprocess the image
        processed_image = player_detection.preprocess_frame(image)
        positions = []
        
        with torch.no_grad():
            results = self.yolo_model(processed_image)  # YOLOv10 inference
            people = []
            
            for result in results:
                boxes = result.boxes.data  # returns boxes object for bbox outputs
                for box in boxes:
                    if box[5] == 0:  # class 0 is person
                        x1, y1, x2, y2, conf, cls = box
                        people.append([int(x1), int(y1), int(x2), int(y2)])
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            for box in people:
                positions.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        return positions
