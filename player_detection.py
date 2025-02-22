import sys
import os
import gc
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov7'))

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import cv2
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.datasets import letterbox
import numpy as np

# Function to identify players using DETR
def identify_players_detr(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.88)[0]

    people = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"DETR detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}")
        if model.config.id2label[label.item()] == 'person':
            people.append(box)
    return people

# Function to identify players using YOLOv7
def identify_players_yolov7(image, model, device):
    with torch.no_grad():  # Disable gradient calculation
        # Preprocess the image for YOLOv7
        img = letterbox(image, new_shape=640)[0]  # Resize and pad the image
        img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert to CHW format
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).float() / 255.0  # Normalize to [0, 1]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Perform inference
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=[0], agnostic=False)  # Class 0 is 'person'

        people = []
        for det in pred:  # detections per image
            if len(det):
                # Rescale boxes to original image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                for *xyxy, conf, cls in det:
                    people.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
        
        # Clear CUDA cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return people

# Load DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Load YOLOv7 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolov7_model = attempt_load("yolov7.pt", map_location=device)
yolov7_model.eval()

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
        yolo_people = identify_players_yolov7(frame, yolov7_model, device)
        
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
    if device.type == 'cuda':
        torch.cuda.empty_cache()

cap.release()
cv2.destroyAllWindows()
