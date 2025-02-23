import base64
import os

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def get_image_messages(play_number):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    first_frame = os.path.join(base_path, f"images/play_{play_number}_map_first_frame.jpg")
    full_map = os.path.join(base_path, f"images/play_{play_number}_map.jpg")
    
    messages = []
    if os.path.exists(first_frame):
        first_frame_base64 = encode_image(first_frame)
        if first_frame_base64:
            messages.append(first_frame_base64)
    
    if os.path.exists(full_map):
        full_map_base64 = encode_image(full_map)
        if full_map_base64:
            messages.append(full_map_base64)
    
    return messages
