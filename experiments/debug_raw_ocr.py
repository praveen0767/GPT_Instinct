import os
import cv2
import sys
import json

# Add project root to path
sys.path.append(os.getcwd())

from ocr_pipeline.trocr_adapter import TrOCRAdapter
from ocr_pipeline.paddle_adapter import PaddleAdapter
from ocr_pipeline.easyocr_adapter import EasyOCRAdapter

def get_raw_ocr(img_path):
    image = cv2.imread(img_path)
    if image is None:
        print("Image not found")
        return

    trocr = TrOCRAdapter()
    paddle = PaddleAdapter()
    easy = EasyOCRAdapter()

    print(f"\n--- RAW OCR RESULTS FOR: {img_path} ---")
    
    print("\n[TrOCR]")
    try:
        res = trocr.recognize(image)
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Error: {e}")

    print("\n[PaddleOCR]")
    try:
        res = paddle.recognize(image)
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Error: {e}")

    print("\n[EasyOCR Raw Detections]")
    try:
        raw_results = easy.reader.readtext(image)
        for r in raw_results:
            print(f"Text: {r[1]}, Conf: {r[2]}")
        res = easy.recognize(image)
        print("\n[EasyOCR Final]")
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    img = "D:/GPT_instinct/examples/debug_session/single_test/test_screenshot_03_sr_enhanced.jpg"
    get_raw_ocr(img)
