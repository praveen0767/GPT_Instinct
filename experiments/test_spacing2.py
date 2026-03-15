import cv2
import glob
from ocr_pipeline.paddle_adapter import PaddleAdapter
from ocr_pipeline.trocr_adapter import TrOCRAdapter

def test_squeezing():
    # Load the exact SR crop from the last test_single_image.py run
    files = glob.glob("debug_artifacts/sr_487b43ef_image.png.png")
    if not files:
        print("Required test file missing...")
        return
        
    img_path = files[0]
    print(f"Testing on {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image")
        return
    
    paddle = PaddleAdapter()
    trocr = TrOCRAdapter()
    
    print("--- Vanilla Inference ---")
    print("Paddle:", paddle.recognize(img)["text"])
    print("TrOCR:", trocr.recognize(img)["text"])
    
    # 1. Squeezing Horizontal
    print("\n--- Squeezed Horizontal (0.5x) ---")
    squeezed = cv2.resize(img, None, fx=0.4, fy=1.0, interpolation=cv2.INTER_AREA)
    cv2.imwrite("debug_test_squeezed_487.png", squeezed)
    print("Paddle:", paddle.recognize(squeezed)["text"])
    print("TrOCR:", trocr.recognize(squeezed)["text"])
    
    # 2. Morphology Dilation Horizontal
    print("\n--- Morphology Dilation Horizontal ---")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Connect horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Invert back to black text on white
    dilated = cv2.bitwise_not(dilated)
    dilated_rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("debug_test_morph_487.png", dilated_rgb)
    
    print("Paddle:", paddle.recognize(dilated_rgb)["text"])
    print("TrOCR:", trocr.recognize(dilated_rgb)["text"])

if __name__ == "__main__":
    test_squeezing()
