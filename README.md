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


models onedong:

wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
