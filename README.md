# INNHABIT
Invation of AAU Innovate by tracking current occupant count




Usage of jetson nano.
Connect via micro-b-cable
ssh nvjetson 192.168.55.1

connect to wifi via
nmcli dev wifi
nmcli dev wifi connect *ssid* -ask

Monitor hardware usage via
jtop

To enable GPGPU support in opencv
OpenCV needs to be build with CUDA & OPENCL using following scripts.  3.5 hours to compile on Jetson Nano
https://github.com/Qengineering/Install-OpenCV-Jetson-Nano


# Models

### coco.names
`wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`

### YoloV3
```sh
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

### YoloV7-tiny
```sh
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7-tiny.cfg
```

## Install OpenCV
`sudo apt install libopencv-dev`

## Compile
```sh
make
```
