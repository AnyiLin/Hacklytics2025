from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import cv2

def identify_players(image, processor, model):

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.88)[0]

    people = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}")
        if model.config.id2label[label.item()] == 'person':
            people.append(box)
    return people



# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


# Open the MP4 file
cap = cv2.VideoCapture('Play 1.mp4')

while True:
    ret, frame = cap.read()
    if not ret:  # Break the loop if there are no frames left
        break

    # Convert the image from BGR to RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a PIL Image from the NumPy array
    pil_image = Image.fromarray(rgb_image)

    people = identify_players(pil_image, processor, model)

    # Draw each bounding box on the image
    for box in people:
        x_min, y_min, x_max, y_max = map(int, box)  # Ensure coordinates are integers
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

    width = frame.shape[1]
    height = frame.shape[0]
    width = int(width / 2)
    height = int(height / 2)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # Display the image with the drawn bounding boxes
    cv2.imshow("Image with Bounding Boxes", frame)
    
    # Exit if the Escape key is pressed
    if cv2.waitKey(0) & 0xFF == 27:
        break
    cv2.waitKey(0)

# Clean up
cap.release()
cv2.destroyAllWindows()
