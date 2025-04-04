#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "detector.h"
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

/**
 * @brief Represents a rectangular bounding box zone.
 *
 * This struct is used to define an area with top-left (x1, y1) and bottom-right (x2, y2) coordinates.
 */
struct BoxZone {
    int x1, y1; ///< Top-left corner coordinates
    int x2, y2; ///< Bottom-right corner coordinates
};

/**
 * @class Configuration
 * @brief Handles application settings loaded from a JSON configuration file.
 *
 * This class manages model settings, device metadata, API configurations, 
 * and entrance zone definitions for object detection.
 */
class Configuration {
public:
    /// Default constructor
    Configuration() = default;

    /**
     * @brief Loads configuration settings from a JSON file.
     *
     * @param configFilePath Path to the configuration JSON file.
     * @return A shared pointer to a Configuration object.
     * @throws std::runtime_error if the configuration file cannot be opened.
     */
    static std::shared_ptr<Configuration> loadFromFile(const std::string& configFilePath);

    /**
     * @brief Updates configuration settings from a JSON object.
     *
     * @param j JSON object containing new settings.
     */
    void updateFromJson(const json& j);

    /**
     * @brief Gets the model file path.
     * @return The model path as a string.
     */
    std::string getModelPath() const { return modelPath_; }

    /**
     * @brief Gets the device name.
     * @return The device name as a string.
     */
    std::string getDeviceName() const { return deviceName_; }

    /**
     * @brief Gets the device location.
     * @return The device location as a string.
     */
    std::string getDeviceLocation() const { return deviceLocation_; }

    /**
     * @brief Gets the server API key.
     * @return The API key as a string.
     */
    std::string getServerApiKey() const { return serverApiKey_; }

    /**
     * @brief Gets the server API base URL.
     * @return The server API URL as a string.
     */
    std::string getServerApi() const { return serverApi_; }

    /**
     * @brief Gets the server entry endpoint.
     * @return The entry endpoint URL as a string.
     */
    std::string getServerEntryEndpoint() const { return serverEntryEndpoint_; }

    /**
     * @brief Gets the server exit endpoint.
     * @return The exit endpoint URL as a string.
     */
    std::string getServerExitEndpoint() const { return serverExitEndpoint_; }

    /**
     * @brief Gets the entrance zones defined in the configuration.
     * @return A vector of BoxZone objects representing the entrance zones.
     */
    const std::vector<BoxZone>& getEntranceZones() const { return entranceZones_; }

private:
    std::string modelPath_; ///< Path to the model file.
    std::string deviceName_; ///< Name of the device.
    std::string deviceLocation_; ///< Location of the device.
    std::string serverApiKey_; ///< API key for server authentication.
    std::string serverApi_; ///< Base URL of the server API.
    std::string serverEntryEndpoint_; ///< Endpoint for entry event logging.
    std::string serverExitEndpoint_; ///< Endpoint for exit event logging.
    std::vector<BoxZone> entranceZones_; ///< List of entrance zones.
};

#endif // COMMON_H
