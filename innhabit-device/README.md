# Object Detection on Orange Pi 5 and Jetson Nano

This project implements object detection using YOLO models on Orange Pi 5 (RK3588 NPU) and NVIDIA Jetson Nano (CUDA), with a portable codebase supporting multiple platforms.


## todo
* save on sd card if requests fail
* systemd Service

## Prerequisites
### rk3588
https://github.com/Qengineering/YoloV5-NPU-Multithread

### jetson
https://github.com/MoussaGRICHE/car_moto_tracking_Jetson_Nano_Yolov8_TensorRT
https://github.com/hamdiboukamcha/Yolo-V11-cpp-TensorRT/tree/master
https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano/tree/main?tab=readme-ov-file
### Hardware Requirements

- **Orange Pi 5**: Install RKNN Toolkit and place `yolov5s-640-640.rknn` in the `models/` directory.
- **Jetson Nano**: Install OpenCV with CUDA support and place `yolov7-tiny.cfg` and `yolov7-tiny.weights` in the `models/` directory.
- **x86\_64**: Install OpenCV:
  ```bash
  sudo apt-get install libopencv-dev
  ```

### Common Requirements

- Ensure `coco.names` is present in the `models/` directory (download from the YOLO repository).

### Downloading Models

#### COCO Labels

```bash
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

#### RKNN Models (NPU)

Pretrained models for RK3588 can be found at [Rockchip's model zoo](https://github.com/airockchip/rknn_model_zoo).

Example:

```bash
wget https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov7/yolov7-tiny.onnx
```

#### NVIDIA Jetson Models (CUDA)

```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7-tiny.cfg
```

---

## Build Instructions

### Default Build (Orange Pi 5)

```bash
chmod +x build-linux.sh
./build-linux.sh
```

### Build Flags

- `-t TARGET`: Set the target platform (`rk3588`, `cuda`, `cpu`)
- `-b BUILD_TYPE`: Set the build type (`Release`, `Debug`)

### Example: Build for NVIDIA Jetson

```bash
./build-linux.sh -t cuda -b Release
```

The executable is located in the `build/bin` directory.

## Running the Program

### Basic Usage

```bash
./object_detection
```

By default, the program captures video from `/dev/video0`.

### Processing Videos and Images

- `--video <file>` : Process a video file
- `--image <file>` : Process a single image

Example: Process a video file

```bash
./object_detection --video horse.mp4
```

---

# Model export


```python
model = YOLO('best.pt')

model.export(
    format="onnx",
    imgsz=512,
    simplify=True,
    half=True,
)
```

```bash
 trtexec --onnx=best.onnx --saveEngine=best512.engine --fp16
```
304.637 qps

## performance
Average timing over 100 frames:
  Preprocess: 0.516046 ms
  Inference: 1.67796 ms
  Postprocess: 1.40035 ms
  Total: 3.59435 ms
Average timing over 100 frames:
  Preprocess: 0.528373 ms
  Inference: 2.21991 ms
  Postprocess: 1.15734 ms
  Total: 3.90562 ms
Average timing over 100 frames:
  Preprocess: 0.534024 ms
  Inference: 2.20707 ms
  Postprocess: 1.17203 ms
  Total: 3.91312 ms
Average timing over 100 frames:
  Preprocess: 0.536733 ms
  Inference: 3.46766 ms
  Postprocess: 0.9452 ms
  Total: 4.94959 ms