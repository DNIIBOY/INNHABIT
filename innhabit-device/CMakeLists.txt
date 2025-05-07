cmake_minimum_required(VERSION 3.10)
project(ObjectDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# Find OpenCV (ensure CUDA support is enabled)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_CUDA_VERSION)
    message(WARNING "OpenCV was not built with CUDA support. GPU acceleration may be limited.")
endif()

# Find CURL
find_package(CURL REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${CURL_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

# Options for platform support
option(USE_CUDA "Enable CUDA support for Jetson" OFF)
option(USE_RKNN "Enable RKNN support for RK3588" OFF)
option(USE_TENSORRT "Enable TensorRT support" OFF)  # Assuming TensorRT is needed

# Enable debug output in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
    message(STATUS "Debug mode enabled with verbose output")
endif()

# Platform-specific configurations
if(USE_CUDA OR USE_TENSORRT)
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    link_directories(${CUDAToolkit_LIBRARY_DIRS})
endif()

if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

if(USE_RKNN)
    add_definitions(-DUSE_RKNN)
    find_library(RKNN_API_LIBRARY rknnrt PATHS /usr/lib /usr/local/lib)
    if(NOT RKNN_API_LIBRARY)
        message(FATAL_ERROR "RKNN API library (librknnrt.so) not found! Please install it.")
    endif()
    include_directories(/usr/local/include)
endif()

if(USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
    find_library(TENSORRT_LIBRARY nvinfer PATHS /usr/lib /usr/local/lib)
    find_library(TENSORRT_ONNX_LIBRARY nvonnxparser PATHS /usr/lib /usr/local/lib)
    find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin PATHS /usr/lib /usr/local/lib)
    if(NOT TENSORRT_LIBRARY OR NOT TENSORRT_ONNX_LIBRARY OR NOT TENSORRT_PLUGIN_LIBRARY)
        message(FATAL_ERROR "TensorRT libraries (libnvinfer.so, libnvonnxparser.so, libnvinfer_plugin.so) not found!")
    endif()
    include_directories(/usr/include/x86_64-linux-gnu)
endif()

# Define common source files
set(SOURCES
    src/core/main.cpp
    src/core/api_handler.cpp
    src/core/cameraThread.cpp
    src/core/detectionThread.cpp
    src/core/detector.cpp
    src/core/devicePoll.cpp
    src/core/displayThread.cpp
    src/core/rtspStreamManager.cpp
    src/core/configuration.cpp
    src/tracker/people_tracker.cpp
    src/tracker/tracked_person.cpp
    src/tracker/utils.cpp
)

# Define platform-specific source files
set(PLATFORM_SOURCES "")
if(USE_RKNN)
    list(APPEND PLATFORM_SOURCES src/platforms/rk3588/rk3588.cpp)
endif()
if(USE_CUDA)
    list(APPEND PLATFORM_SOURCES src/platforms/cuda/cuda.cpp)
endif()
if(USE_TENSORRT)
    list(APPEND PLATFORM_SOURCES src/platforms/tensorrt/tensorrt.cpp)
    set(CUDA_KERNEL_SOURCES src/platforms/tensorrt/preprocess.cu)
endif()
if(NOT USE_CUDA AND NOT USE_RKNN AND NOT USE_TENSORRT)
    list(APPEND PLATFORM_SOURCES src/platforms/cpu/cpu.cpp)
endif()

# Compile CUDA kernels if TensorRT is enabled
if(USE_TENSORRT)
    cuda_add_library(cuda_kernels STATIC ${CUDA_KERNEL_SOURCES})
    set_target_properties(cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Add executable
add_executable(object_detection ${SOURCES} ${PLATFORM_SOURCES})

# Link libraries
target_link_libraries(object_detection PRIVATE 
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
)

if(USE_CUDA OR USE_TENSORRT)
    find_library(CUDART_LIBRARY cudart HINTS ${CUDAToolkit_LIBRARY_DIRS} PATHS /usr/local/cuda/lib64)
    if(NOT CUDART_LIBRARY)
        message(FATAL_ERROR "CUDA runtime library (libcudart.so) not found!")
    endif()
    target_link_libraries(object_detection PRIVATE ${CUDART_LIBRARY})
endif()

if(USE_RKNN)
    target_link_libraries(object_detection PRIVATE ${RKNN_API_LIBRARY})
endif()

if(USE_TENSORRT)
    target_link_libraries(object_detection PRIVATE 
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNX_LIBRARY}
        ${TENSORRT_PLUGIN_LIBRARY}
        cuda_kernels
    )
endif()

# Set output directory
set_target_properties(object_detection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Print configuration status
message(STATUS "Building ObjectDetection with:")
message(STATUS "  - OpenCV: ${OpenCV_VERSION} (CUDA: ${OpenCV_CUDA_VERSION})")
message(STATUS "  - CURL: ${CURL_VERSION_STRING}")
message(STATUS "  - USE_CUDA: ${USE_CUDA}")
message(STATUS "  - USE_RKNN: ${USE_RKNN}")
message(STATUS "  - USE_TENSORRT: ${USE_TENSORRT}")
message(STATUS "  - Platform sources: ${PLATFORM_SOURCES}")
message(STATUS "  - CUDA Kernel sources: ${CUDA_KERNEL_SOURCES}")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - CUDA Library: ${CUDART_LIBRARY}")