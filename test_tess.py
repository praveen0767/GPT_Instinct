import cv2
import glob
import pytesseract

def test_tesseract():
    files = glob.glob("debug_artifacts/sr_487*-03-05*")
    if not files:
        files = glob.glob("debug_artifacts/sr_487b43ef_image.png.png")
    
    if not files:
        print("Required test file missing...")
        return
        
    img_path = files[0]
    print(f"Testing Tesseract on {img_path}")
    
    img = cv2.imread(img_path)
    
    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    print("\n--- Tesseract Vanilla (PSM 6: Uniform block of text) ---")
    text = pytesseract.image_to_string(img, config='--psm 6 -c tessedit_char_whitelist=0123456789.')
    print(f"[{text.strip()}]")
    
    print("\n--- Tesseract Threshold (PSM 6) ---")
    text2 = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789.')
    print(f"[{text2.strip()}]")
    
    print("\n--- Tesseract Threshold (PSM 7: Single line) ---")
    text3 = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789.')
    print(f"[{text3.strip()}]")

if __name__ == "__main__":
    test_tesseract()
