# Object Detection on Orange Pi 5 and Jetson Nano

This project implements object detection using YOLO models on ONVIDIA Jetson Nano (CUDA) and Orange Pi 5 (RK3588 NPU) with a portable codebase supporting multiple platforms.


## Prerequisites
### jetson
https://github.com/MoussaGRICHE/car_moto_tracking_Jetson_Nano_Yolov8_TensorRT
https://github.com/hamdiboukamcha/Yolo-V11-cpp-TensorRT/tree/master
https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano/tree/main?tab=readme-ov-file

### rk3588
https://github.com/Qengineering/YoloV5-NPU-Multithread

#### RKNN Models (NPU)


## Build Instructions

### Default Build (Orange Pi 5)

```bash
chmod +x build-linux.sh
./build-linux.sh
```

### Build Flags

- `-t TARGET`: Set the target platform (`rk3588`, `cuda`, `cpu`, `tensorrt`)
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
# Systemd setup

## frpc.service

```sh
[Unit]
Description=FRP client
After=network.target

[Install]
WantedBy=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/frp/frpc -c /usr/local/frp/frpc.toml
Restart=always
RestartSec=5
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=%n
```

## frpc.toml

```toml
serverAddr = "{{ .Envs.FRPS_SERVER_ADDR }}"
serverPort = "{{ .Envs.FRPS_BIND_PORT }}"

[[proxies]]
name = "ssh1"
type = "tcpmux"
multiplexer = "httpconnect"
customDomains = ["{{ .Envs.FRPC_DOMAIN }}"]
localIP = "127.0.0.1"
localPort = 22
```

## innhabit.service
[Unit]
Description=innhabit service
After=network.target

[Service]
ExecStart=/home/nvjetson/innhabit-device/build/object_detection
Restart=on-failure
WorkingDirectory=/home/nvjetson/innhabit-device/build/
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
