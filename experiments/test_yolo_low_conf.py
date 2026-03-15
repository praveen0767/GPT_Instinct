from ultralytics import YOLO

def test_yolo():
    try:
        model = YOLO(r'D:\GPT_instinct\models\yolov8_detector.pt')
        results = model(r'D:\GPT_instinct\meter_test.png', conf=0.01)
        print("--- DETECTIONS ---")
        for r in results:
            for box in r.boxes:
                print(f"Class: {int(box.cls[0])}, Conf: {float(box.conf[0]):.4f}, Box: {box.xyxy[0].tolist()}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_yolo()
