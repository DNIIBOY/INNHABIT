#include "configuration.h"
#include <fstream>
#include <stdexcept>

std::shared_ptr<Configuration> Configuration::GetInstance() {
    static std::shared_ptr<Configuration> instance = std::make_shared<Configuration>();
    return instance;
}

std::shared_ptr<Configuration> Configuration::LoadFromFile(const std::string& config_file_path) {
    auto instance = GetInstance();
    instance->LoadConfigFile(config_file_path);
    return instance;
}

void Configuration::LoadConfigFile(const std::string& config_file_path) {
    std::unique_lock<std::shared_mutex> write_lock(mutex_);
    std::ifstream file(config_file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_file_path);
    }
    nlohmann::json json_data;
    try {
        file >> json_data;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("Failed to parse JSON: " + std::string(e.what()));
    }
    UpdateFromJson(json_data);
}

void Configuration::UpdateFromJson(const nlohmann::json& json_data) {
    std::unique_lock<std::shared_mutex> write_lock(mutex_);
    if (json_data.contains("model_path")) {
        model_path_ = json_data["model_path"].get<std::string>();
    }
    if (json_data.contains("device_name")) {
        device_name_ = json_data["device_name"].get<std::string>();
    }
    if (json_data.contains("device_location")) {
        device_location_ = json_data["device_location"].get<std::string>();
    }
    if (json_data.contains("server_api_key")) {
        server_api_key_ = json_data["server_api_key"].get<std::string>();
    }
    if (json_data.contains("server_api")) {
        server_api_ = json_data["server_api"].get<std::string>();
    }
    if (json_data.contains("server_entry_endpoint")) {
        server_entry_endpoint_ = json_data["server_entry_endpoint"].get<std::string>();
    }
    if (json_data.contains("server_exit_endpoint")) {
        server_exit_endpoint_ = json_data["server_exit_endpoint"].get<std::string>();
    }
    if (json_data.contains("zones")) {
        typed_zones_.clear();
        for (const auto& zone : json_data["zones"]) {
            int x1 = zone["x1"].get<int>();
            int y1 = zone["y1"].get<int>();
            int x2 = zone["x2"].get<int>();
            int y2 = zone["y2"].get<int>();
            std::string type = zone["type"].get<std::string>();
            tracker::ZoneType zone_type = type == "ENTRY" ? tracker::ZoneType::ENTRY : tracker::ZoneType::EXIT;
            typed_zones_.emplace_back(x1, y1, x2, y2, zone_type);
        }
    }
    // Update config_data_ for custom key-value pairs
    if (json_data.contains("custom")) {
        for (const auto& [key, value] : json_data["custom"].items()) {
            if (value.is_string()) {
                config_data_[key] = value.get<std::string>();
            } else if (value.is_number_integer()) {
                config_data_[key] = value.get<int>();
            } else if (value.is_number_float()) {
                config_data_[key] = value.get<double>();
            } else if (value.is_boolean()) {
                config_data_[key] = value.get<bool>();
            }
        }
    }
}

bool Configuration::SaveToFile(const std::string& file_path) const {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    nlohmann::json json_data;
    json_data["model_path"] = model_path_;
    json_data["device_name"] = device_name_;
    json_data["device_location"] = device_location_;
    json_data["server_api_key"] = server_api_key_;
    json_data["server_api"] = server_api_;
    json_data["server_entry_endpoint"] = server_entry_endpoint_;
    json_data["server_exit_endpoint"] = server_exit_endpoint_;
    json_data["zones"] = nlohmann::json::array();
    for (const auto& zone : typed_zones_) {
        nlohmann::json zone_json;
        zone_json["x1"] = zone.zone.x1;
        zone_json["y1"] = zone.zone.y1;
        zone_json["x2"] = zone.zone.x2;
        zone_json["y2"] = zone.zone.y2;
        zone_json["type"] = zone.type == tracker::ZoneType::ENTRY ? "ENTRY" : "EXIT";
        json_data["zones"].push_back(zone_json);
    }
    json_data["custom"] = nlohmann::json::object();
    for (const auto& [key, value] : config_data_) {
        std::visit([&](const auto& val) { json_data["custom"][key] = val; }, value);
    }

    std::ofstream file(file_path);
    if (!file.is_open()) {
        return false;
    }
    file << json_data.dump(4);
    return true;
}

std::string Configuration::GetModelPath() const {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    return model_path_;
}

std::string Configuration::GetDeviceName() const {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    return device_name_;
}

std::string Configuration::GetDeviceLocation() const {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    return device_location_;
}

std::string Configuration::GetServerApiKey() const {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    return server_api_key_;
}

std::string Configuration::GetServerApi() const {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    return server_api_;
}

std::string Configuration::GetServerEntryEndpoint() const {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    return server_entry_endpoint_;
}

std::string Configuration::GetServerExitEndpoint() const {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    return server_exit_endpoint_;
}

std::vector<tracker::TypedZone> Configuration::GetTypedZones() const {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    return typed_zones_;
}

bool Configuration::HasKey(const std::string& key) const {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    return config_data_.find(key) != config_data_.end();
}

bool Configuration::RemoveKey(const std::string& key) {
    std::unique_lock<std::shared_mutex> write_lock(mutex_);
    return config_data_.erase(key) > 0;
}