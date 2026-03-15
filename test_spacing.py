import cv2
import glob
from ocr_pipeline.paddle_adapter import PaddleAdapter
from ocr_pipeline.trocr_adapter import TrOCRAdapter

def test_squeezing():
    files = glob.glob("debug_artifacts/crop_*Screenshot 2026-03-05 154657.png.png")
    if not files:
        files = glob.glob("debug_artifacts/sr_*Screenshot 2026-03-05 154657.png.png")
    
    if not files:
        print("Required test file missing...")
        return
        
    img_path = files[0]
    print(f"Testing on {img_path}")
    
    img = cv2.imread(img_path)
    
    paddle = PaddleAdapter()
    trocr = TrOCRAdapter()
    
    print("--- Vanilla Inference ---")
    print("Paddle:", paddle.recognize(img)["text"])
    print("TrOCR:", trocr.recognize(img)["text"])
    
    # Let's try squeezing
    print("\n--- Squeezed Horizontal (0.5x) ---")
    squeezed = cv2.resize(img, None, fx=0.5, fy=1.0, interpolation=cv2.INTER_AREA)
    cv2.imwrite("debug_test_squeezed.png", squeezed)
    print("Paddle:", paddle.recognize(squeezed)["text"])
    print("TrOCR:", trocr.recognize(squeezed)["text"])
    
    # Let's try extreme squeezing
    print("\n--- Extreme Squeezed Horizontal (0.3x) ---")
    squeezed2 = cv2.resize(img, None, fx=0.3, fy=1.0, interpolation=cv2.INTER_AREA)
    print("Paddle:", paddle.recognize(squeezed2)["text"])
    print("TrOCR:", trocr.recognize(squeezed2)["text"])

    # Let's try morphology dilation (closing gaps)
    print("\n--- Morphology Dilation Horizontal ---")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binary inverse
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    # Invert back to black text on white
    dilated = cv2.bitwise_not(dilated)
    # Convert to 3 channels for models
    dilated_rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    print("Paddle:", paddle.recognize(dilated_rgb)["text"])
    print("TrOCR:", trocr.recognize(dilated_rgb)["text"])

if __name__ == "__main__":
    test_squeezing()
