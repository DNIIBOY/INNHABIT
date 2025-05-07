#!/bin/bash

# Default values
TARGET="cpu"
BUILD_TYPE="Release"
CUDA_PATH="/usr/local/cuda"

# Function to display usage
usage() {
    echo "Usage: $0 [-t TARGET] [-b BUILD_TYPE] [-c CUDA_PATH]"
    echo "  -t TARGET: Set the target platform (cpu, cuda, tensorrt, rk3588)"
    echo "  -b BUILD_TYPE: Set the build type (Release, Debug)"
    echo "  -c CUDA_PATH: Set the CUDA toolkit path (default: /usr/local/cuda)"
    echo "Example: $0 -t tensorrt -b Debug -c /usr/local/cuda-11.8"
    exit 1
}

# Parse command-line arguments
while getopts "t:b:c:h" opt; do
    case $opt in
        t) TARGET="$OPTARG";;
        b) BUILD_TYPE="$OPTARG";;
        c) CUDA_PATH="$OPTARG";;
        h) usage;;
        ?) usage;;
    esac
done

# Validate inputs
if [[ "$BUILD_TYPE" != "Release" && "$BUILD_TYPE" != "Debug" ]]; then
    echo "Error: BUILD_TYPE must be 'Release' or 'Debug'"
    usage
fi

if [[ "$TARGET" != "cpu" && "$TARGET" != "cuda" && "$TARGET" != "tensorrt" && "$TARGET" != "rk3588" ]]; then
    echo "Error: Invalid TARGET. Valid options: cpu, cuda, tensorrt, rk3588"
    usage
fi

# Check dependencies
echo "Checking dependencies..."

# OpenCV
if ! pkg-config --exists opencv4; then
    echo "Error: OpenCV not found. Install with 'sudo apt-get install libopencv-dev'."
    exit 1
fi
echo "OpenCV found: $(pkg-config --modversion opencv4)"

# CURL
if ! pkg-config --exists libcurl; then
    echo "Error: CURL not found. Install with 'sudo apt-get install libcurl4-openssl-dev'."
    exit 1
fi
echo "CURL found"

# CUDA (for cuda or tensorrt targets)
if [[ "$TARGET" == "cuda" || "$TARGET" == "tensorrt" ]]; then
    if [[ ! -d "$CUDA_PATH" || ! -f "$CUDA_PATH/bin/nvcc" ]]; then
        echo "Error: CUDA not found at $CUDA_PATH. Install CUDA or specify correct path with -c."
        exit 1
    fi
    echo "CUDA found at $CUDA_PATH"
fi

# TensorRT (for tensorrt target)
if [[ "$TARGET" == "tensorrt" ]]; then
    if ! ldconfig -p | grep -q libnvinfer.so; then
        echo "Error: TensorRT not found. Install TensorRT for your architecture."
        exit 1
    fi
    echo "TensorRT found"
fi

# RKNN (for rk3588 target)
if [[ "$TARGET" == "rk3588" ]]; then
    if ! ldconfig -p | grep -q librknnrt.so; then
        echo "Error: RKNN API library (librknnrt.so) not found. Install RKNN toolkit."
        exit 1
    fi
    echo "RKNN found"
fi

# Set up build directory
BUILD_DIR="build"
if [[ ! -d "$BUILD_DIR" ]]; then
    mkdir -p "$BUILD_DIR"
fi
cd "$BUILD_DIR"

# Configure CMake
echo "Configuring CMake for $TARGET ($BUILD_TYPE)..."
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
if [[ "$TARGET" == "rk3588" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DUSE_RKNN=ON"
elif [[ "$TARGET" == "cuda" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH"
elif [[ "$TARGET" == "tensorrt" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DUSE_TENSORRT=ON -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH"
fi

cmake $CMAKE_ARGS ..
if [[ $? -ne 0 ]]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

# Build
echo "Building with $(nproc) jobs..."
make -j$(nproc)
if [[ $? -ne 0 ]]; then
    echo "Error: Build failed. Check errors above."
    exit 1
fi

echo "Build completed successfully!"
echo "Target: $TARGET, Build Type: $BUILD_TYPE"
echo "Run './object_detection' for camera or './object_detection --image sample.jpg' for an image."