import cv2
import numpy as np
import easyocr
import os

def test_preprocessing(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Standard Inversion (LCDs often work better inverted)
    inverted = cv2.bitwise_not(gray)
    
    # 2. Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 3. CLAHE + Inversion
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    cl_inv = cv2.bitwise_not(cl)

    reader = easyocr.Reader(['en'], gpu=False)
    
    print("--- Testing Preprocessing ---")
    print(f"Original: {reader.readtext(gray, allowlist='0123456789')}")
    print(f"Inverted: {reader.readtext(inverted, allowlist='0123456789')}")
    print(f"Thresh: {reader.readtext(thresh, allowlist='0123456789')}")
    print(f"CLAHE Inverted: {reader.readtext(cl_inv, allowlist='0123456789')}")

if __name__ == "__main__":
    img = "D:/GPT_instinct/examples/debug_session/single_test/test_screenshot_03_sr_enhanced.jpg"
    test_preprocessing(img)
