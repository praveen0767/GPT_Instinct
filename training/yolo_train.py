import os
from ultralytics import YOLO

def train_yolo_detector():
    """
    Step 5 & 7 of Master Prompt V3:
    Trains a unified detection model (yolov8n) to detect:
    - Display bounding explicit region
    - Serial Nameplate explicitly
    - Individual digits 0-9 for segmentation
    """
    print("Initializing YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt")
    
    data_yaml = r"D:\GPT_instinct\dataset\yolov8_data.yaml"
    
    print(f"Starting training on dataset {data_yaml}...")
    results = model.train(
        data=data_yaml,
        epochs=10,  # Training for 10 epochs per user request (Approx 2.5 hours)
        imgsz=640,
        batch=16,
        name="agm_meter_detector",
        device=0,  # Force GPU usage
        amp=True,  # Automatic Mixed Precision to save VRAM on the 6GB RTX 3050
        workers=0, # Fixes Windows WinError 1455 Paging File Too Small crash
        plots=False,
    )
    print("Training loop complete!")
    
    export_path = model.export(format="torchscript")
    print(f"Model exported to {export_path}")

if __name__ == "__main__":
    train_yolo_detector()
