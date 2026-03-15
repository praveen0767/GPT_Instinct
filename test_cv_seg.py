import cv2
import numpy as np

def extract_digits_cv2(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilate slightly to connect segmented digit pieces
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    thresh = cv2.dilate(thresh, kernel_dil, iterations=1)
    
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digits = []
    # Identify digits based on bounding box constraints
    h_img, w_img = img.shape[:2]
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h)
        area = w * h
        
        # Valid digits: reasonably tall, width isn't excessively huge
        if h > h_img * 0.15 and area > 100:
            if aspect > 0.15 and aspect < 1.0:
                digits.append((x, y, w, h))
            
    digits = sorted(digits, key=lambda b: b[0])
    print(f"Found {len(digits)} potential digits")
    for b in digits:
        print(b)
        
    return digits

if __name__ == "__main__":
    import sys
    import glob
    # Find the pre-processed display crops built by test_single_image
    files = glob.glob("debug_artifacts/sr_*.png")
    if not files:
        print("No sr_ files found")
        # try crop_
        files = glob.glob("debug_artifacts/crop_*.png")
        if not files:
            files = ["Screenshot 2026-03-05 154657.png"]
            
    test_img = files[0]
    print(f"Testing on {test_img}")
    extract_digits_cv2(test_img)
