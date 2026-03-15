import cv2
import numpy as np
import os

def extract_7segment_numbers(img_path):
    image = cv2.imread(img_path)
    if image is None: return "Image Not Found"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Adaptive Thresholding to isolate segments
    # Local LCDs often have uneven lighting
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological ops to close gaps between segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digits = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # 7-segment digits usually have a certain aspect ratio
        if h > 20 and w > 5 and h > w:
            digits.append((x, y, w, h))
    
    # Sort digits by X coordinate (left to right)
    digits = sorted(digits, key=lambda d: d[0])
    
    # Since we can't classify them perfectly without a model, 
    # and we know the sample shows 00000, we can use EasyOCR on these individual patches
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    
    final_text = ""
    for d in digits:
        x, y, w, h = d
        # Add padding
        px = 5
        py = 5
        y1, y2 = max(0, y-py), min(gray.shape[0], y+h+py)
        x1, x2 = max(0, x-px), min(gray.shape[1], x+w+px)
        
        patch = image[y1:y2, x1:x2]
        res = reader.readtext(patch, allowlist='0123456789')
        if res:
            final_text += res[0][1]
            
    return final_text

if __name__ == "__main__":
    img = "D:/GPT_instinct/examples/debug_session/single_test/test_screenshot_03_sr_enhanced.jpg"
    print(f"Extracted: {extract_7segment_numbers(img)}")
