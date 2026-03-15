import cv2
import numpy as np

def main():
    img = cv2.imread(r'D:\GPT_instinct\Screenshot 2026-03-14 014904.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Find LCD Screen
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    sx, sy, sw, sh = cv2.boundingRect(c)
    
    # Pad LCD slightly
    px = int(sw * 0.05); py = int(sh * 0.1)
    sx1 = max(0, sx - px); sy1 = max(0, sy - py)
    sx2 = min(img.shape[1], sx + sw + px); sy2 = min(img.shape[0], sy + sh + py)
    
    lcd = img[sy1:sy2, sx1:sx2]
    
    # 2. Find black digits inside LCD
    lcd_hsv = cv2.cvtColor(lcd, cv2.COLOR_BGR2HSV)
    # Define black threshold
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])
    digit_mask = cv2.inRange(lcd_hsv, lower_black, upper_black)
    
    # Morph to bridge digit gaps (7-segment gaps)
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    digit_mask = cv2.dilate(digit_mask, kernel_d, iterations=2)
    
    dcnts, _ = cv2.findContours(digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter constraints
    boxes = []
    for cnt in dcnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if 50 < w * h < 4000: # Sanity check for digits on this size screen
            # Convert back to absolute image coordinates
            abs_x = sx1 + x
            abs_y = sy1 + y
            boxes.append((abs_x, abs_y, w, h))
            
    # Sort left to right
    boxes = sorted(boxes, key=lambda b: b[0])
    
    img_h, img_w = img.shape[:2]
    
    print(f"FOUND {len(boxes)} POSSIBLE DIGITS!")
    
    # We want exactly 7 boxes (1,2,3,4,5,.,6). If we find more/less, we map them directly for YOLO script.
    for i, (bx, by, bw, bh) in enumerate(boxes):
        cx = (bx + bw/2) / img_w
        cy = (by + bh/2) / img_h
        nw = bw / img_w
        nh = bh / img_h
        print(f"Index {i}: YOLO box: {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} | Abs: {bx},{by},{bw},{bh} (Area={bw*bh})")

if __name__ == '__main__':
    main()
