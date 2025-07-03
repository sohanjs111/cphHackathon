import base64
import io
from PIL import Image
import numpy as np
from typing import List
import cv2

def process_vision_info(messages):
    """
    Extracts image and video information from messages.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        tuple: (image_inputs, video_inputs)
    """
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            continue
        
        for item in content:
            if item.get("type") == "image":
                image_data = item.get("image")
                if isinstance(image_data, str):
                    # If it's a path, open the image
                    if image_data.startswith("http://") or image_data.startswith("https://"):
                        # Handle URLs (not implemented here)
                        pass
                    else:
                        # Assume it's a local file path
                        try:
                            img = Image.open(image_data)
                            image_inputs.append(img)
                        except Exception as e:
                            print(f"Error loading image: {e}")
                elif isinstance(image_data, (Image.Image, np.ndarray)):
                    # If it's already an image object
                    if isinstance(image_data, np.ndarray):
                        image_data = Image.fromarray(image_data)
                    image_inputs.append(image_data)
                    
            elif item.get("type") == "video":
                video_data = item.get("video")
                if video_data is not None:
                    video_inputs.append(video_data)
    
    return image_inputs, video_inputs

