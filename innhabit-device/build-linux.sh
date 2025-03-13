#!/bin/bash

# Default values
TARGET="cpu"
BUILD_TYPE="Release"

# Function to display usage
usage() {
    echo "Usage: $0 [-t TARGET] [-b BUILD_TYPE]"
    echo "  -t TARGET: Set the target platform (rk3588, cuda, tensorrt, cpu)"
    echo "  -b BUILD_TYPE: Set the build type (Release, Debug)"
    echo "Example: $0 -t tensorrt -b Debug"
    exit 1
}

# Parse command-line arguments
while getopts "t:b:h" opt; do
    case $opt in
        t) TARGET="$OPTARG";;
        b) BUILD_TYPE="$OPTARG";;
        h) usage;;
        ?) usage;;
    esac
done

# Validate build type
if [[ "$BUILD_TYPE" != "Release" && "$BUILD_TYPE" != "Debug" ]]; then
    echo "Error: BUILD_TYPE must be 'Release' or 'Debug'"
    usage
fi

# Check for OpenCV
if ! pkg-config --exists opencv4; then
    echo "Error: OpenCV not found. Install with 'sudo apt-get install libopencv-dev'."
    exit 1
fi

# Check for TensorRT if targeting tensorrt
if [ "$TARGET" = "tensorrt" ]; then
    if ! ldconfig -p | grep -q libnvinfer.so; then
        echo "Error: TensorRT not found. Install TensorRT for your architecture."
        exit 1
    fi
fi

# Set up build directory
BUILD_DIR="build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake based on target
case $TARGET in
    "rk3588")
        echo "Configuring for RK3588 with RKNN..."
        cmake -DUSE_RKNN=ON -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
        ;;
    "cuda")
        echo "Configuring for CUDA (OpenCV DNN)..."
        cmake -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
        ;;
    "tensorrt")
        echo "Configuring for TensorRT..."
        cmake -DUSE_TENSORRT=ON -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
        ;;
    "cpu")
        echo "Configuring for generic CPU..."
        cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
        ;;
    *)
        echo "Error: Unsupported target: $TARGET"
        echo "Valid targets: rk3588, cuda, tensorrt, cpu"
        exit 1
        ;;
esac

# Build
make -j$(nproc)
if [ $? -eq 0 ]; then
    echo "Build completed successfully (Type: $BUILD_TYPE, Target: $TARGET)."
    echo "Run './object_detection' for camera or './object_detection --image sample.jpg' for an image."
else
    echo "Build failed. Check errors above."
    exit 1
fi