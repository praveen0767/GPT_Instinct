from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='yolo_data/data.yaml', epochs=20, imgsz=640, batch=8, name='meter_field_detector')
