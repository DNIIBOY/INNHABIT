#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "common.h" // Include first to define LOG and BoxZone
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <mutex>
#include <shared_mutex>
#include <fstream>
#include <memory>
#include <optional>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * @class Configuration
 * @brief Thread-safe configuration class that can be shared between threads
 *
 * This class manages application settings loaded from a JSON configuration file
 * and provides thread-safe access to these settings. It uses a read-write lock
 * (shared_mutex) to allow multiple concurrent reads but exclusive writes.
 */
class Configuration : public std::enable_shared_from_this<Configuration> {
public:
    /**
     * @brief Get the singleton instance of the Configuration class
     * @return A shared pointer to the Configuration instance
     */
    static std::shared_ptr<Configuration> getInstance() {
        static std::shared_ptr<Configuration> instance = nullptr;
        static std::mutex instanceMutex;

        std::lock_guard<std::mutex> lock(instanceMutex);
        if (!instance) {
            instance = std::shared_ptr<Configuration>(new Configuration());
        }
        return instance;
    }

    /**
     * @brief Loads configuration settings from a JSON file
     * @param configFilePath Path to the configuration JSON file
     * @return A shared pointer to the Configuration object
     * @throws std::runtime_error if the configuration file cannot be opened
     */
    static std::shared_ptr<Configuration> loadFromFile(const std::string& configFilePath) {
        auto config = getInstance();
        config->loadConfigFile(configFilePath);
        return config;
    }

    /**
     * @brief Updates configuration settings from a JSON object
     * @param j JSON object containing new settings
     */
    void updateFromJson(const json& j) {
        std::unique_lock<std::shared_mutex> writeLock(m_mutex);

        // Update standard fields if present
        if (j.contains("ModelPath")) m_modelPath = j["ModelPath"].get<std::string>();
        if (j.contains("DeviceName")) m_deviceName = j["DeviceName"].get<std::string>();
        if (j.contains("DeviceLocation")) m_deviceLocation = j["DeviceLocation"].get<std::string>();
        if (j.contains("ServerApiKey")) m_serverApiKey = j["ServerApiKey"].get<std::string>();
        if (j.contains("ServerAPI")) m_serverApi = j["ServerAPI"].get<std::string>();
        if (j.contains("ServerEntryEndpoint")) m_serverEntryEndpoint = j["ServerEntryEndpoint"].get<std::string>();
        if (j.contains("ServerExitEndpoint")) m_serverExitEndpoint = j["ServerExitEndpoint"].get<std::string>();

        // Process entry and exit boxes
        m_entranceZones.clear(); // Clear existing zones to avoid duplicates
        if (j.contains("entry_box")) {
            if (!j["entry_box"].is_array() || j["entry_box"].size() != 4) {
                ERROR("Invalid entry_box format: Expected array of 4 integers");
            } else {
                try {
                    auto zoneArray = j["entry_box"].get<std::vector<int>>();
                    BoxZone zone = {zoneArray[0], zoneArray[1], zoneArray[2], zoneArray[3]};
                    if (zone.x1 < zone.x2 && zone.y1 < zone.y2) { // Basic validation
                        m_entranceZones.push_back(zone);
                        LOG("Updated entry_box: (x1=" << zone.x1 << ", y1=" << zone.y1 
                            << ", x2=" << zone.x2 << ", y2=" << zone.y2 << ")");
                    } else {
                        ERROR("Invalid entry_box coordinates: x1 >= x2 or y1 >= y2");
                    }
                } catch (const json::exception& e) {
                    ERROR("Failed to parse entry_box: " << e.what());
                }
            }
        }

        if (j.contains("exit_box")) {
            if (!j["exit_box"].is_array() || j["exit_box"].size() != 4) {
                ERROR("Invalid exit_box format: Expected array of 4 integers");
            } else {
                try {
                    auto zoneArray = j["exit_box"].get<std::vector<int>>();
                    BoxZone zone = {zoneArray[0], zoneArray[1], zoneArray[2], zoneArray[3]};
                    if (zone.x1 < zone.x2 && zone.y1 < zone.y2) { // Basic validation
                        m_entranceZones.push_back(zone);
                        LOG("Updated exit_box: (x1=" << zone.x1 << ", y1=" << zone.y1 
                            << ", x2=" << zone.x2 << ", y2=" << zone.y2 << ")");
                    } else {
                        ERROR("Invalid exit_box coordinates: x1 >= x2 or y1 >= y2");
                    }
                } catch (const json::exception& e) {
                    ERROR("Failed to parse exit_box: " << e.what());
                }
            }
        }

        // Update custom settings
        for (auto it = j.begin(); it != j.end(); ++it) {
            const std::string& key = it.key();
            if (key == "ModelPath" || key == "DeviceName" || key == "DeviceLocation" ||
                key == "ServerApiKey" || key == "ServerAPI" ||
                key == "ServerEntryEndpoint" || key == "ServerExitEndpoint" ||
                key == "entry_box" || key == "exit_box") {
                continue;
            }

            if (it->is_string()) {
                m_configData[key] = it->get<std::string>();
            } else if (it->is_number_integer()) {
                m_configData[key] = it->get<int>();
            } else if (it->is_number_float()) {
                m_configData[key] = it->get<double>();
            } else if (it->is_boolean()) {
                m_configData[key] = it->get<bool>();
            }
        }

        LOG("Configuration updated from JSON with " << m_entranceZones.size() << " zones");
    }

    /**
     * @brief Save the current configuration to a JSON file
     * @param filePath Path where the configuration file should be saved
     * @return True if saving was successful, false otherwise
     */
    bool saveToFile(const std::string& filePath) const {
        std::shared_lock<std::shared_mutex> readLock(m_mutex);

        try {
            json j;
            j["ModelPath"] = m_modelPath;
            j["DeviceName"] = m_deviceName;
            j["DeviceLocation"] = m_deviceLocation;
            j["ServerApiKey"] = m_serverApiKey;
            j["ServerAPI"] = m_serverApi;
            j["ServerEntryEndpoint"] = m_serverEntryEndpoint;
            j["ServerExitEndpoint"] = m_serverExitEndpoint;

            // Save entry_box and exit_box explicitly
            if (!m_entranceZones.empty()) {
                if (m_entranceZones.size() >= 1) {
                    j["entry_box"] = {m_entranceZones[0].x1, m_entranceZones[0].y1, 
                                     m_entranceZones[0].x2, m_entranceZones[0].y2};
                }
                if (m_entranceZones.size() >= 2) {
                    j["exit_box"] = {m_entranceZones[1].x1, m_entranceZones[1].y1, 
                                    m_entranceZones[1].x2, m_entranceZones[1].y2};
                }
            }

            for (const auto& [key, value] : m_configData) {
                try {
                    if (std::holds_alternative<std::string>(value)) {
                        j[key] = std::get<std::string>(value);
                    } else if (std::holds_alternative<int>(value)) {
                        j[key] = std::get<int>(value);
                    } else if (std::holds_alternative<double>(value)) {
                        j[key] = std::get<double>(value);
                    } else if (std::holds_alternative<bool>(value)) {
                        j[key] = std::get<bool>(value);
                    }
                } catch (const std::bad_variant_access&) {
                    ERROR("Failed to convert config value for key: " << key);
                }
            }

            std::ofstream file(filePath);
            if (!file.is_open()) {
                ERROR("Failed to open file for saving configuration: " << filePath);
                return false;
            }

            file << j.dump(4);
            file.close();
            LOG("Configuration saved to: " << filePath);
            return true;
        } catch (const std::exception& e) {
            ERROR("Error saving configuration: " << e.what());
            return false;
        }
    }

    // Getters for standard configuration properties (thread-safe)
    std::string getModelPath() const {
        std::shared_lock<std::shared_mutex> readLock(m_mutex);
        return m_modelPath;
    }

    std::string getDeviceName() const {
        std::shared_lock<std::shared_mutex> readLock(m_mutex);
        return m_deviceName;
    }

    std::string getDeviceLocation() const {
        std::shared_lock<std::shared_mutex> readLock(m_mutex);
        return m_deviceLocation;
    }

    std::string getServerApiKey() const {
        std::shared_lock<std::shared_mutex> readLock(m_mutex);
        return m_serverApiKey;
    }

    std::string getServerApi() const {
        std::shared_lock<std::shared_mutex> readLock(m_mutex);
        return m_serverApi;
    }

    std::string getServerEntryEndpoint() const {
        std::shared_lock<std::shared_mutex> readLock(m_mutex);
        return m_serverEntryEndpoint;
    }

    std::string getServerExitEndpoint() const {
        std::shared_lock<std::shared_mutex> readLock(m_mutex);
        return m_serverExitEndpoint;
    }

    std::vector<BoxZone> getEntranceZones() const {
        std::shared_lock<std::shared_mutex> readLock(m_mutex);
        return m_entranceZones;
    }

    // Generic value getter with type conversion (thread-safe)
    template<typename T>
    std::optional<T> getValue(const std::string& key) const {
        std::shared_lock<std::shared_mutex> readLock(m_mutex);

        auto it = m_configData.find(key);
        if (it != m_configData.end()) {
            try {
                return std::get<T>(it->second);
            } catch (const std::bad_variant_access&) {
                return std::nullopt;
            }
        }
        return std::nullopt;
    }

    // Generic value setter (thread-safe)
    template<typename T>
    void setValue(const std::string& key, const T& value) {
        std::unique_lock<std::shared_mutex> writeLock(m_mutex);
        m_configData[key] = value;
    }

    // Check if a key exists (thread-safe)
    bool hasKey(const std::string& key) const {
        std::shared_lock<std::shared_mutex> readLock(m_mutex);
        return m_configData.find(key) != m_configData.end();
    }

    // Remove a key (thread-safe)
    bool removeKey(const std::string& key) {
        std::unique_lock<std::shared_mutex> writeLock(m_mutex);
        return m_configData.erase(key) > 0;
    }

private:
    Configuration() = default;

    Configuration(const Configuration&) = delete;
    Configuration& operator=(const Configuration&) = delete;
    Configuration(Configuration&&) = delete;
    Configuration& operator=(Configuration&&) = delete;

    void loadConfigFile(const std::string& configFilePath) {
        std::unique_lock<std::shared_mutex> writeLock(m_mutex);

        std::ifstream file(configFilePath);
        if (!file.is_open()) {
            ERROR("Failed to open config file: " << configFilePath);
            throw std::runtime_error("Config file not found");
        }

        json j;
        try {
            file >> j;
        } catch (const json::exception& e) {
            ERROR("Failed to parse config file " << configFilePath << ": " << e.what());
            throw std::runtime_error("Invalid JSON format in config file");
        }

        m_modelPath = j.value("ModelPath", std::string());
        m_deviceName = j.value("DeviceName", std::string());
        m_deviceLocation = j.value("DeviceLocation", std::string());
        m_serverApiKey = j.value("ServerApiKey", std::string());
        m_serverApi = j.value("ServerAPI", std::string());
        m_serverEntryEndpoint = j.value("ServerEntryEndpoint", std::string());
        m_serverExitEndpoint = j.value("ServerExitEndpoint", std::string());

        // Process entry and exit boxes
        m_entranceZones.clear();
        if (j.contains("entry_box")) {
            if (!j["entry_box"].is_array() || j["entry_box"].size() != 4) {
                ERROR("Invalid entry_box format in file: Expected array of 4 integers");
            } else {
                try {
                    auto zoneArray = j["entry_box"].get<std::vector<int>>();
                    BoxZone zone = {zoneArray[0], zoneArray[1], zoneArray[2], zoneArray[3]};
                    if (zone.x1 < zone.x2 && zone.y1 < zone.y2) {
                        m_entranceZones.push_back(zone);
                        LOG("Loaded entry_box: (x1=" << zone.x1 << ", y1=" << zone.y1 
                            << ", x2=" << zone.x2 << ", y2=" << zone.y2 << ")");
                    } else {
                        ERROR("Invalid entry_box coordinates in file: x1 >= x2 or y1 >= y2");
                    }
                } catch (const json::exception& e) {
                    ERROR("Failed to parse entry_box from file: " << e.what());
                }
            }
        }

        if (j.contains("exit_box")) {
            if (!j["exit_box"].is_array() || j["exit_box"].size() != 4) {
                ERROR("Invalid exit_box format in file: Expected array of 4 integers");
            } else {
                try {
                    auto zoneArray = j["exit_box"].get<std::vector<int>>();
                    BoxZone zone = {zoneArray[0], zoneArray[1], zoneArray[2], zoneArray[3]};
                    if (zone.x1 < zone.x2 && zone.y1 < zone.y2) {
                        m_entranceZones.push_back(zone);
                        LOG("Loaded exit_box: (x1=" << zone.x1 << ", y1=" << zone.y1 
                            << ", x2=" << zone.x2 << ", y2=" << zone.y2 << ")");
                    } else {
                        ERROR("Invalid exit_box coordinates in file: x1 >= x2 or y1 >= y2");
                    }
                } catch (const json::exception& e) {
                    ERROR("Failed to parse exit_box from file: " << e.what());
                }
            }
        }

        // Load additional custom settings
        for (auto it = j.begin(); it != j.end(); ++it) {
            const std::string& key = it.key();
            if (key == "ModelPath" || key == "DeviceName" || key == "DeviceLocation" ||
                key == "ServerApiKey" || key == "ServerAPI" ||
                key == "ServerEntryEndpoint" || key == "ServerExitEndpoint" ||
                key == "entry_box" || key == "exit_box") {
                continue;
            }

            if (it->is_string()) {
                m_configData[key] = it->get<std::string>();
            } else if (it->is_number_integer()) {
                m_configData[key] = it->get<int>();
            } else if (it->is_number_float()) {
                m_configData[key] = it->get<double>();
            } else if (it->is_boolean()) {
                m_configData[key] = it->get<bool>();
            }
        }

        LOG("Configuration loaded from file: " << configFilePath << " with " << m_entranceZones.size() << " zones");
    }

    mutable std::shared_mutex m_mutex;
    std::string m_modelPath;
    std::string m_deviceName;
    std::string m_deviceLocation;
    std::string m_serverApiKey;
    std::string m_serverApi;
    std::string m_serverEntryEndpoint;
    std::string m_serverExitEndpoint;
    std::vector<BoxZone> m_entranceZones;
    std::unordered_map<std::string, std::variant<std::string, int, double, bool>> m_configData;
};

#endif // CONFIGURATION_H