from ultralytics import YOLO

def main():
    # Path to your YOLOv8 base weights and data.yaml
    model_name = "yolov8n.pt"  # nano, good for fast training/prototype
    data_yaml = r"D:\GPT\test\data.yaml"

    # Load base model
    model = YOLO(model_name)

    # Train
    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=8,            # change if you have more/less VRAM
        name="meter_field_detector",
        project=r"D:\GPT\runs",   # where to save logs/weights
        verbose=True
    )

if __name__ == "__main__":
    main()
