from ultralytics import YOLO

def main():
    model = YOLO(r'D:\GPT_instinct\models\yolov8_detector.pt')
    results = model(r'D:\GPT_instinct\Screenshot 2026-03-14 014904.png', conf=0.01)
    names = model.names
    print("--- DETECTED CLASSES ON NEW METER ---")
    for box in results[0].boxes:
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        print(f"ClassID {cls_id} -> '{names[cls_id]}' (Conf: {conf:.4f}) at {box.xyxy[0].tolist()}")

if __name__ == '__main__':
    main()
