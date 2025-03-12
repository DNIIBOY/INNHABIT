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
# If common.h is in src/, add: ${CMAKE_SOURCE_DIR}/src
# If common.h is in src/include/, add: ${CMAKE_SOURCE_DIR}/src/include

# Options for platform support
option(USE_CUDA "Enable CUDA support for Jetson" OFF)
option(USE_RKNN "Enable RKNN support for RK3588" OFF)

# Enable debug output in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
    message(STATUS "Debug mode enabled with verbose output")
endif()

# Platform-specific configurations
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})
endif()

if(USE_RKNN)
    add_definitions(-DUSE_RKNN)
    find_library(RKNN_API_LIBRARY rknnrt PATHS /usr/lib /usr/local/lib)
    if(NOT RKNN_API_LIBRARY)
        message(FATAL_ERROR "RKNN API library (librknnrt.so) not found! Please install it.")
    endif()
    include_directories(/usr/local/include)
endif()

# Define common source files
set(COMMON_SOURCES
    src/main.cpp
    src/detector.cpp
    src/postprocess.cpp
    src/tracker.cpp
)

# Define platform-specific source files
set(PLATFORM_SOURCES "")
if(USE_RKNN)
    list(APPEND PLATFORM_SOURCES src/platforms/rk3588.cpp)
endif()
if(USE_CUDA)
    list(APPEND PLATFORM_SOURCES src/platforms/jetson.cpp)
endif()
if(NOT USE_CUDA AND NOT USE_RKNN)
    list(APPEND PLATFORM_SOURCES src/platforms/cpu.cpp)
    # Always include CPU support as fallback/default
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

if(USE_CUDA)
    target_link_libraries(object_detection PRIVATE ${CUDA_LIBRARIES} cudart)
endif()

if(USE_RKNN)
    target_link_libraries(object_detection PRIVATE ${RKNN_API_LIBRARY})
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
message(STATUS "  - Platform sources: ${PLATFORM_SOURCES}")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
