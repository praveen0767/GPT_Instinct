from ultralytics import YOLO
model = YOLO(r'D:\GPT_instinct\models\yolov8_detector.pt')
print('YOLO CLASSES:', model.names)
