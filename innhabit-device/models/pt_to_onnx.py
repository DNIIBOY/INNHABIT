from ultralytics import YOLO

model = YOLO("best.pt")
model.export(
        format="onnx",
        imgsz=512,
        simplify=True,
        half=True
    )
