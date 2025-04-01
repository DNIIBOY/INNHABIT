#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "detector.h"
#include <nlohmann/json.hpp>

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

/**
* Labels for COCO
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

// Structure for a box zone (top-left and bottom-right coordinates)
/**
* this is for boxed
    */
struct BoxZone {
    int x1, y1;  // Top-left
    int x2, y2;  // Bottom-right
};

class Configuration {
    public:
        Configuration() = default;
    
        // Load initial settings from JSON file
        static std::shared_ptr<Configuration> loadFromFile(const std::string& configFilePath) {
            std::ifstream file(configFilePath);
            if (!file.is_open()) {
                ERROR("Failed to open config file: " << configFilePath);
                throw std::runtime_error("Config file not found");
            }
    
            json j;
            file >> j;
            auto config = std::make_shared<Configuration>();
    
            config->modelPath_ = j["ModelPath"].get<std::string>();
            config->deviceName_ = j["DeviceName"].get<std::string>();
            config->deviceLocation_ = j["DeviceLokation"].get<std::string>();
            config->serverApiKey_ = j["ServerApiKey"].get<std::string>();
            config->serverApi_ = j["ServerAPI"].get<std::string>();
            config->serverEntryEndpoint_ = j["ServerEntryEndpoint"].get<std::string>();
            config->serverExitEndpoint_ = j["ServerExitEndpoint"].get<std::string>();
    
            auto entranceZone = j["EntranceZone"];
            for (auto it = entranceZone.begin(); it != entranceZone.end(); ++it) {
                auto zoneArray = it.value().get<std::vector<int>>();
                if (zoneArray.size() == 4) {
                    BoxZone zone = {zoneArray[0], zoneArray[1], zoneArray[2], zoneArray[3]};
                    config->entranceZones_.push_back(zone);
                } else {
                    ERROR("Invalid zone format in config: " << it.key());
                }
            }
    
            return config;
        }
    
        // Update settings from JSON (e.g., from API response)
        void updateFromJson(const json& j) {
            if (j.contains("ModelPath")) modelPath_ = j["ModelPath"].get<std::string>();
            if (j.contains("DeviceName")) deviceName_ = j["DeviceName"].get<std::string>();
            if (j.contains("DeviceLokation")) deviceLocation_ = j["DeviceLokation"].get<std::string>();
            if (j.contains("ServerApiKey")) serverApiKey_ = j["ServerApiKey"].get<std::string>();
            if (j.contains("ServerAPI")) serverApi_ = j["ServerAPI"].get<std::string>();
            if (j.contains("ServerEntryEndpoint")) serverEntryEndpoint_ = j["ServerEntryEndpoint"].get<std::string>();
            if (j.contains("ServerExitEndpoint")) serverExitEndpoint_ = j["ServerExitEndpoint"].get<std::string>();
    
            if (j.contains("EntranceZone")) {
                entranceZones_.clear();
                auto entranceZone = j["EntranceZone"];
                for (auto it = entranceZone.begin(); it != entranceZone.end(); ++it) {
                    auto zoneArray = it.value().get<std::vector<int>>();
                    if (zoneArray.size() == 4) {
                        BoxZone zone = {zoneArray[0], zoneArray[1], zoneArray[2], zoneArray[3]};
                        entranceZones_.push_back(zone);
                    } else {
                        ERROR("Invalid zone format in config update: " << it.key());
                    }
                }
            }
        }
    
        // Getters
        std::string getModelPath() const { return modelPath_; }
        std::string getDeviceName() const { return deviceName_; }
        std::string getDeviceLocation() const { return deviceLocation_; }
        std::string getServerApiKey() const { return serverApiKey_; }
        std::string getServerApi() const { return serverApi_; }
        std::string getServerEntryEndpoint() const { return serverEntryEndpoint_; }
        std::string getServerExitEndpoint() const { return serverExitEndpoint_; }
        const std::vector<BoxZone>& getEntranceZones() const { return entranceZones_; }
    
    private:
        std::string modelPath_;
        std::string deviceName_;
        std::string deviceLocation_;
        std::string serverApiKey_;
        std::string serverApi_;
        std::string serverEntryEndpoint_;
        std::string serverExitEndpoint_;
        std::vector<BoxZone> entranceZones_;
    };
    
#endif // COMMON_H
