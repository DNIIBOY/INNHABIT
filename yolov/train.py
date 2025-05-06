import sys
from ultralytics import YOLO
import ultralytics

# Check environment
ultralytics.checks()

# Use the first command-line argument as the model path (default to "yolov8n.pt" if not provided)
model_path = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"

# Load the model from the specified path
model = YOLO(model_path)

# Run initial validation (optional)
initmetrics = model.val()
print("Initial validation metrics:", initmetrics)

model.train(
    data="dataset/data.yaml", 
    epochs=100,  
    imgsz=512,  
    batch=32,  
    patience=10,  
    lr0=0.0001,  
    lrf=0.00005,  
    cos_lr=True,  
    optimizer="SGD",  
    momentum=0.94,  
    weight_decay=0.0005,  
    hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,  
    flipud=0.5, fliplr=0.5,  
    degrees=15,  
    translate=0.2, scale=0.7, shear=0.2,  
    mosaic=1.0, mixup=0.5,  
    iou=0.19,  
    conf=0.09,  
)

# Run validation after training
metrics = model.val()
print("Initial validation metrics:", initmetrics)
print("Post-training validation metrics:", metrics)
