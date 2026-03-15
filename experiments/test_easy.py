import cv2
import glob
import easyocr

def test_easyocr():
    # Load exact API output
    files = glob.glob("debug_artifacts/sr_487b43ef_image.png.png")
    if not files:
        print("Required test file missing...")
        return
        
    img_path = files[0]
    print(f"Testing EasyOCR on {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image")
        return
        
    reader = easyocr.Reader(['en'], gpu=False)
    
    print("\n--- EasyOCR Vanilla ---")
    res1 = reader.readtext(img, allowlist='0123456789.')
    print([r[1] for r in res1])
    
    # 1. Squeezing Horizontal
    print("\n--- Squeezed Horizontal (0.4x) ---")
    squeezed = cv2.resize(img, None, fx=0.4, fy=1.0, interpolation=cv2.INTER_AREA)
    cv2.imwrite("debug_test_easy_squeezed.png", squeezed)
    res2 = reader.readtext(squeezed, allowlist='0123456789.')
    print([r[1] for r in res2])
    
    # 2. Morphology Dilation Horizontal
    print("\n--- Morphology Dilation Horizontal ---")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Connect horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Invert back to black text on white
    dilated = cv2.bitwise_not(dilated)
    cv2.imwrite("debug_test_easy_morph.png", dilated)
    
    res3 = reader.readtext(dilated, allowlist='0123456789.')
    print([r[1] for r in res3])

if __name__ == "__main__":
    test_easyocr()
