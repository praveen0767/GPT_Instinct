import cv2
import glob
import numpy as np
import easyocr
import os

from ocr_pipeline.paddle_adapter import PaddleAdapter
from ocr_pipeline.trocr_adapter import TrOCRAdapter

def test_dot_matrix():
    files = glob.glob("debug_artifacts/sr_487b43ef_image.png.png")
    if not files: return
    img = cv2.imread(files[0])
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold to isolate digits (white on black)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    
    # 1. Connect dot matrix (Dilation)
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    solid = cv2.dilate(thresh, kernel_connect, iterations=1)
    
    # 2. Smooth rough edges (Morphology Close)
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel_smooth)
    
    # Invert back to black text on white background (OCR models prefer this)
    clean_inv = cv2.bitwise_not(clean)
    clean_rgb = cv2.cvtColor(clean_inv, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("debug_test_solid.png", clean_rgb)
    
    # 3. Horizontal Squeeze (to fix wide spacing)
    squeezed = cv2.resize(clean_rgb, None, fx=0.5, fy=1.0, interpolation=cv2.INTER_AREA)
    cv2.imwrite("debug_test_solid_squeezed.png", squeezed)
    
    print("Models loading...")
    paddle = PaddleAdapter()
    trocr = TrOCRAdapter()
    reader = easyocr.Reader(['en'], gpu=False)
    
    print("\n--- Solid Font (No Squeeze) ---")
    print("Paddle:", paddle.recognize(clean_rgb)["text"])
    print("TrOCR:", trocr.recognize(clean_rgb)["text"])
    print("EasyOCR:", [r[1] for r in reader.readtext(clean_rgb, allowlist='0123456789.')])
    
    print("\n--- Solid Font + Squeezed ---")
    print("Paddle:", paddle.recognize(squeezed)["text"])
    print("TrOCR:", trocr.recognize(squeezed)["text"])
    print("EasyOCR:", [r[1] for r in reader.readtext(squeezed, allowlist='0123456789.')])

if __name__ == "__main__":
    test_dot_matrix()
