#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "detector.h"
#include <nlohmann/json.hpp>

using namespace std;
using namespace cv;
using json = nlohmann::json;

// Detection threshold constants
#ifndef BOX_THRESH
#define BOX_THRESH 0.20
#endif
#ifndef NMS_THRESH
#define NMS_THRESH 0.45
#endif

// Logging helpers
#define LOG(msg) cout << "[INFO] " << msg << endl
#define ERROR(msg) cerr << "[ERROR] " << msg << endl

// COCO class labels
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

// Structure for a box zone (top-left and bottom-right coordinates)
struct BoxZone {
    int x1, y1;  // Top-left
    int x2, y2;  // Bottom-right
};

// Structure for configuration
struct Config {
    std::string modelEnginePath;
    std::string deviceName;
    std::string deviceLokation;
    std::string serverIP;
    std::string serverPort;
    std::string serverEventAPI;
    std::string serverEntryEndpoint;
    std::string serverExitEndpoint;
    std::vector<BoxZone> entranceZones;  // Vector of box zones
};

// Utility function to load config from JSON
inline Config loadConfig(const std::string& configFilePath) {
    Config config;
    std::ifstream file(configFilePath);
    if (!file.is_open()) {
        ERROR("Failed to open config file: " << configFilePath);
        throw std::runtime_error("Config file not found");
    }

    json j;
    file >> j;

    config.modelEnginePath = j["ModelEnginePath"].get<std::string>();
    config.deviceName = j["DeviceName"].get<std::string>();
    config.serverIP = j["ServerIP"].get<std::string>();
    config.serverPort = j["ServerPort"].get<std::string>();
    config.serverEventAPI = j["ServerEventAPI"].get<std::string>();
    config.serverEntryEndpoint = j["ServerEntryEndpoint"].get<std::string>();
    config.serverExitEndpoint = j["ServerExitEndpoint"].get<std::string>();


    auto entranceZone = j["EntranceZone"];
    for (auto it = entranceZone.begin(); it != entranceZone.end(); ++it) {
        auto zoneArray = it.value().get<std::vector<int>>();
        if (zoneArray.size() == 4) {
            BoxZone zone = {zoneArray[0], zoneArray[1], zoneArray[2], zoneArray[3]};
            config.entranceZones.push_back(zone);
        } else {
            ERROR("Invalid zone format in config: " << it.key());
        }
    }

    return config;
}

#endif // COMMON_H