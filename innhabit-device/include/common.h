#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <tracker/zone_type.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/// Detection threshold for object detection
#ifndef BOX_THRESH
#define BOX_THRESH 0.20 ///< Confidence threshold for object detection
#endif

#ifndef NMS_THRESH
#define NMS_THRESH 0.45 ///< Non-Maximum Suppression (NMS) threshold
#endif

/// Logging macros
#define LOG(msg) std::cout << "[INFO] " << msg << std::endl ///< Logs an informational message
#define ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl ///< Logs an error message

/**
 * @brief Labels for the COCO dataset.
 *
 * This array contains the class labels for the COCO dataset, used for object detection.
 */
static const char* const COCO_LABELS[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

// Include the new Configuration class
#include "configuration.h"

#endif // COMMON_H
