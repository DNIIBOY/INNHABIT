#!/bin/bash

# Default values
TARGET="rk3588"
BUILD_TYPE="Release"

# Function to display usage
usage() {
    echo "Usage: $0 [-t TARGET] [-b BUILD_TYPE]"
    echo "  -t TARGET: Set the target platform (rk3588, cuda, cpu)"
    echo "  -b BUILD_TYPE: Set the build type (Release, Debug)"
    echo "Example: $0 -t rk3588 -b Debug"
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

# Detect platform if not specified
if [ -z "$TARGET" ]; then
    PLATFORM=$(uname -m)
    echo "Target not specified, auto-detecting platform: $PLATFORM"
    if [ "$PLATFORM" = "x86_64" ] || [ "$PLATFORM" = "aarch64" ]; then
        if [ -f /usr/lib/librknnrt.so ] || [ -f /usr/local/lib/librknnrt.so ]; then
            TARGET="rk3588"
        elif [ -d /usr/local/cuda ]; then
            TARGET="cuda"
        else
            TARGET="cpu"
        fi
    else
        echo "Unsupported platform: $PLATFORM"
        exit 1
    fi
else
    echo "Target specified: $TARGET"
fi

# Check for OpenCV
if ! pkg-config --exists opencv4; then
    echo "Error: OpenCV not found. Install with 'sudo apt-get install libopencv-dev'."
    exit 1
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
        echo "Configuring for Jetson with CUDA..."
        cmake -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
        ;;
    "cpu")
        echo "Configuring for generic CPU..."
        cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
        ;;
    *)
        echo "Error: Unsupported target: $TARGET"
        echo "Valid targets: rk3588, cuda, cpu"
        exit 1
        ;;
esac

# Build
make -j$(nproc)
if [ $? -eq 0 ]; then
    echo "Build complete (Type: $BUILD_TYPE, Target: $TARGET)."
    echo "Run './bin/object_detection' for camera or './bin/object_detection --image' for sample.jpg."
else
    echo "Build failed. Check errors above."
    exit 1
fi
