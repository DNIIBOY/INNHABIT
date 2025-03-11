# Object Detection on Orange Pi 5 and Jetson Nano

This project implements object detection using YOLO models on Orange Pi 5 (RK3588 NPU) and NVIDIA Jetson Nano (CUDA), with a portable codebase supporting multiple platforms.

## Prerequisites

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

# Code Overview

## Directory Structure

```
.
├── include/
│   ├── detector.h
│   ├── postprocess.h
│   └── tracker.h
├── models/
├── src/
│   ├── platforms/
│   │   ├── cpu.cpp         # Optimized for x86_64
│   │   ├── jetson.cpp      # Optimized for Jetson Nano (CUDA)
│   │   └── rk3588.cpp     # Optimized for RK3588 (NPU)
│   ├── detector.cpp       # Platform-independent wrapper
│   ├── tracker.cpp        # Tracks detected objects
│   ├── postprocess.cpp    # Post-processes detection outputs
│   └── main.cpp           # Main entry point
└── build/
```

## Key Components

### `detector.cpp`

- Provides a unified interface for object detection.
- Uses platform-specific implementations based on the target hardware.

### `tracker.cpp`

- Implements object tracking for detected entities.

### `postprocess.cpp`

- Handles post-processing of raw YOLO model outputs (e.g., bounding box parsing).

### `main.cpp`

- Main entry point combining detection and tracking.

---


# Debug

- Ensure all required model files are placed in the `models/` directory.
- Use the appropriate target platform flag when building.
- For additional optimizations, adjust platform-specific code in the `src/platforms` directory.


