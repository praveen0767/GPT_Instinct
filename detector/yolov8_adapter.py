import cv2
import numpy as np

class YOLOv8Adapter:
    """Wrapper for the YOLOv8 detector using the local dataset."""
    def __init__(self, model_path='yolov8n.pt'):
        self.model_path = model_path
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.mock_mode = False
        except (ImportError, Exception):
            print(f"Warning: Could not load YOLOv8 model at {model_path}. Running adapter in mock mode.")
            self.model = None
            self.mock_mode = True

    def detect(self, image: np.ndarray, conf=0.15):
        """Detects meter display and serial number fields in the image."""
        if self.mock_mode:
            # Fallback mock logic for the structural pipeline:
            h, w = image.shape[:2]
            return [
                {"bbox": [int(w*0.2), int(h*0.3), int(w*0.8), int(h*0.5)], "class": "display", "confidence": 0.85},
                {"bbox": [int(w*0.3), int(h*0.7), int(w*0.7), int(h*0.85)], "class": "serial", "confidence": 0.90}
            ]
        
        results = self.model(image, conf=conf)
        detections = []
        has_display = False
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_idx = int(box.cls[0].cpu().numpy())
                if cls_idx == 0:
                    cls_name = "display"
                elif cls_idx == 1:
                    cls_name = "serial"
                elif cls_idx == 12:
                    cls_name = "."
                else:
                    cls_name = str(cls_idx) # '2' -> '2', '10' -> '10'
                
                if cls_name == "display":
                    has_display = True
                conf_score = float(box.conf[0].cpu().numpy())
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class": cls_name,
                    "confidence": conf_score
                })
        
        # DETERMINISTIC FALLBACK: If AI missed the display, physically find the green LCD
        if not has_display:
            try:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # Visiontek green screen thresholds
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    if w > 30 and h > 15:
                        pad_x = int(w * 0.05)
                        pad_y = int(h * 0.1)
                        img_h, img_w = image.shape[:2]
                        x1 = max(0, x - pad_x)
                        y1 = max(0, y - pad_y)
                        x2 = min(img_w, x + w + pad_x)
                        y2 = min(img_h, y + h + pad_y)
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "class": "display",
                            "confidence": 0.88 # Synthetic high confidence to pass to CNN
                        })
            except Exception as e:
                print(f"Fallback LCD detection failed: {e}")
                
        return detections

    def crop_detected_fields(self, image: np.ndarray, detections: list):
        """Returns cropped numpy arrays for each detection dict."""
        crops = {}
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            crop = image[y1:y2, x1:x2]
            crops[det['class']] = crop
        return crops
