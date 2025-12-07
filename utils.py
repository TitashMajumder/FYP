import cv2
import numpy as np
import base64

def crop_and_encode(image_path, x, y, w, h):
    # Load image
    img = cv2.imread(image_path)
    
    # Crop the diseased section
    # ensure coordinates are within image bounds
    if img is not None:
        cropped = img[y:y+h, x:x+w]
        
        # Encode to base64 string for easy transport to React
        _, buffer = cv2.imencode('.jpg', cropped)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"
    return None