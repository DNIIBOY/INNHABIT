cmake_minimum_required(VERSION 3.10)
project(ObjectDetection)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
# Find CURL
find_package(CURL REQUIRED)

# Include directories (assuming common.h is in include/)
include_directories(${OpenCV_INCLUDE_DIRS} ${CURL_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

# Options for platform support
option(USE_CUDA "Enable CUDA support for Jetson" OFF)
option(USE_RKNN "Enable RKNN support for RK3588" OFF)
option(USE_TENSORRT "Enable TensorRT support" OFF)

# Enable debug output in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
    message(STATUS "Debug mode enabled with verbose output")
endif()

# Platform-specific configurations
if(USE_CUDA OR USE_TENSORRT)  # CUDA is required for both CUDA and TensorRT targets
    find_package(CUDA REQUIRED)
    if(NOT CUDA_FOUND)
        message(FATAL_ERROR "CUDA is required for USE_CUDA or USE_TENSORRT but was not found!")
    endif()
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})
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
    if(NOT TENSORRT_LIBRARY)
        message(FATAL_ERROR "TensorRT library (libnvinfer.so) not found! Please install it.")
    endif()
    find_library(TENSORRT_ONNX_LIBRARY nvonnxparser PATHS /usr/lib /usr/local/lib)
    if(NOT TENSORRT_ONNX_LIBRARY)
        message(FATAL_ERROR "TensorRT ONNX parser library (libnvonnxparser.so) not found! Please install it.")
    endif()
    include_directories(/usr/include/x86_64-linux-gnu)  # Adjust if TensorRT headers are elsewhere
endif()

# Define common source files
set(COMMON_SOURCES
    src/main.cpp
    src/detector.cpp
    src/tracker.cpp
    src/api_handler.cpp
)

# Define platform-specific source files
set(PLATFORM_SOURCES "")
if(USE_RKNN)
    list(APPEND PLATFORM_SOURCES src/platforms/rk3588.cpp)
endif()
if(USE_CUDA)
    list(APPEND PLATFORM_SOURCES src/platforms/cuda.cpp)
endif()
if(USE_TENSORRT)
    list(APPEND PLATFORM_SOURCES src/platforms/jetson_tensorrt.cpp)
endif()
if(NOT USE_CUDA AND NOT USE_RKNN AND NOT USE_TENSORRT)
    list(APPEND PLATFORM_SOURCES src/platforms/cpu.cpp)  # Default to CPU if no other target
endif()

# Add executable
add_executable(object_detection
    ${COMMON_SOURCES}
    ${PLATFORM_SOURCES}
)

# Link libraries
target_link_libraries(object_detection PRIVATE 
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
)

if(USE_CUDA OR USE_TENSORRT)
    # Explicitly link CUDA runtime library
    find_library(CUDART_LIBRARY cudart HINTS ${CUDA_LIBRARY_DIRS})
    if(NOT CUDART_LIBRARY)
        message(FATAL_ERROR "CUDA runtime library (libcudart.so) not found!")
    endif()
    target_link_libraries(object_detection PRIVATE ${CUDART_LIBRARY})
endif()

if(USE_RKNN)
    target_link_libraries(object_detection PRIVATE ${RKNN_API_LIBRARY})
endif()

if(USE_TENSORRT)
    target_link_libraries(object_detection PRIVATE ${TENSORRT_LIBRARY} ${TENSORRT_ONNX_LIBRARY})
endif()

# Set output directory
set_target_properties(object_detection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Print configuration status
message(STATUS "Building ObjectDetection with:")
message(STATUS "  - OpenCV: ${OpenCV_VERSION}")
message(STATUS "  - CURL: ${CURL_VERSION_STRING}")
message(STATUS "  - USE_CUDA: ${USE_CUDA}")
message(STATUS "  - USE_RKNN: ${USE_RKNN}")
message(STATUS "  - USE_TENSORRT: ${USE_TENSORRT}")
message(STATUS "  - Platform sources: ${PLATFORM_SOURCES}")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - CUDA Library: ${CUDART_LIBRARY}")