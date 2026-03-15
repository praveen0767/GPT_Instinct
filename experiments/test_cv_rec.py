import cv2
import glob
import os
import easyocr

def extract_and_read():
    files = glob.glob("debug_artifacts/sr_487b43ef_image.png.png")
    if not files:
        print("Required test file missing...")
        return
        
    img_path = files[0]
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    thresh = cv2.dilate(thresh, kernel_dil, iterations=1)
    
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img = img.shape[:2]
    digits = []
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h)
        area = w * h
        if h > h_img * 0.15 and area > 100 and 0.15 < aspect < 1.0:
            digits.append((x, y, w, h))
            
    digits = sorted(digits, key=lambda b: b[0])
    
    reader = easyocr.Reader(['en'], gpu=False)
    
    print(f"Testing on {len(digits)} segmented digit crops:")
    final_text = ""
    os.makedirs("debug_chips", exist_ok=True)
    
    for i, (x, y, w, h) in enumerate(digits):
        # Add padding
        px, py = int(w * 0.2), int(h * 0.2)
        y1, y2 = max(0, y-py), min(h_img, y+h+py)
        x1, x2 = max(0, x-px), min(w_img, x+w+px)
        
        chip = img[y1:y2, x1:x2]
        if chip.size == 0: continue
        cv2.imwrite(f"debug_chips/chip_{i}.png", chip)
        
        # We process the chip. We resize it to be larger so EasyOCR doesn't skip it
        chip_large = cv2.resize(chip, (60, 100), interpolation=cv2.INTER_CUBIC)
        
        res = reader.readtext(chip_large, allowlist='0123456789', min_size=5)
        text = res[0][1] if res else "?"
        print(f"Chip {i} (x={x}): OCR={text}")
        if text.isdigit():
            final_text += text
            
    print(f"\nFinal Assembled Register: {final_text}")

if __name__ == "__main__":
    extract_and_read()
