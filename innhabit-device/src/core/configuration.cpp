#include "configuration.h"

#include <fstream>
#include <stdexcept>

#include "common.h"
#include <tracker/zone_type.h>

namespace {

std::shared_ptr<Configuration> CreateInstance() {
  return std::shared_ptr<Configuration>(new Configuration());
}

}  // namespace

std::shared_ptr<Configuration> Configuration::GetInstance() {
  static std::shared_ptr<Configuration> instance = nullptr;
  static std::mutex instance_mutex;

  std::lock_guard<std::mutex> lock(instance_mutex);
  if (!instance) {
    instance = CreateInstance();
  }
  return instance;
}

std::shared_ptr<Configuration> Configuration::LoadFromFile(
    const std::string& config_file_path) {
  auto config = GetInstance();
  config->LoadConfigFile(config_file_path);
  return config;
}

void Configuration::UpdateFromJson(const nlohmann::json& json_data) {
  std::unique_lock<std::shared_mutex> write_lock(mutex_);

  // Update standard fields if present.
  if (json_data.contains("ModelPath")) {
    model_path_ = json_data["ModelPath"].get<std::string>();
  }
  if (json_data.contains("DeviceName")) {
    device_name_ = json_data["DeviceName"].get<std::string>();
  }
  if (json_data.contains("DeviceLocation")) {
    device_location_ = json_data["DeviceLocation"].get<std::string>();
  }
  if (json_data.contains("ServerApiKey")) {
    server_api_key_ = json_data["ServerApiKey"].get<std::string>();
  }
  if (json_data.contains("ServerAPI")) {
    server_api_ = json_data["ServerAPI"].get<std::string>();
  }
  if (json_data.contains("ServerEntryEndpoint")) {
    server_entry_endpoint_ = json_data["ServerEntryEndpoint"].get<std::string>();
  }
  if (json_data.contains("ServerExitEndpoint")) {
    server_exit_endpoint_ = json_data["ServerExitEndpoint"].get<std::string>();
  }

  // Process entry and exit zones.
  typed_zones_.clear();  // Clear existing zones to avoid duplicates.
  if (json_data.contains("entry_box")) {
    if (!json_data["entry_box"].is_array() ||
        json_data["entry_box"].size() != 4) {
      ERROR("Invalid entry_box format: Expected array of 4 integers [x1, y1, x2, y2]");
    } else {
      try {
        auto zone_array = json_data["entry_box"].get<std::vector<int>>();
        tracker::BoxZone zone = {zone_array[0], zone_array[1], zone_array[2],
                                 zone_array[3]};
        if (zone.x1 < zone.x2 && zone.y1 < zone.y2) {
          typed_zones_.emplace_back(zone, tracker::ZoneType::ENTRY);
          LOG("Updated entry zone: (x1=" << zone.x1 << ", y1=" << zone.y1
                                        << ", x2=" << zone.x2 << ", y2=" << zone.y2
                                        << ")");
        } else {
          ERROR("Invalid entry_box coordinates: x1 >= x2 or y1 >= y2");
        }
      } catch (const nlohmann::json::exception& e) {
        ERROR("Failed to parse entry_box: " << e.what());
      }
    }
  }

  if (json_data.contains("exit_box")) {
    if (!json_data["exit_box"].is_array() ||
        json_data["exit_box"].size() != 4) {
      ERROR("Invalid exit_box format: Expected array of 4 integers [x1, y1, x2, y2]");
    } else {
      try {
        auto zone_array = json_data["exit_box"].get<std::vector<int>>();
        tracker::BoxZone zone = {zone_array[0], zone_array[1], zone_array[2],
                                 zone_array[3]};
        if (zone.x1 < zone.x2 && zone.y1 < zone.y2) {
          typed_zones_.emplace_back(zone, tracker::ZoneType::EXIT);
          LOG("Updated exit zone: (x1=" << zone.x1 << ", y1=" << zone.y1
                                       << ", x2=" << zone.x2 << ", y2=" << zone.y2
                                       << ")");
        } else {
          ERROR("Invalid exit_box coordinates: x1 >= x2 or y1 >= y2");
        }
      } catch (const nlohmann::json::exception& e) {
        ERROR("Failed to parse exit_box: " << e.what());
      }
    }
  }

  // Update custom settings.
  for (auto it = json_data.begin(); it != json_data.end(); ++it) {
    const std::string& key = it.key();
    if (key == "ModelPath" || key == "DeviceName" || key == "DeviceLocation" ||
        key == "ServerApiKey" || key == "ServerAPI" ||
        key == "ServerEntryEndpoint" || key == "ServerExitEndpoint" ||
        key == "entry_box" || key == "exit_box") {
      continue;
    }

    if (it->is_string()) {
      config_data_[key] = it->get<std::string>();
    } else if (it->is_number_integer()) {
      config_data_[key] = it->get<int>();
    } else if (it->is_number_float()) {
      config_data_[key] = it->get<double>();
    } else if (it->is_boolean()) {
      config_data_[key] = it->get<bool>();
    }
  }

  LOG("Configuration updated from JSON with " << typed_zones_.size() << " zones");
}

bool Configuration::SaveToFile(const std::string& file_path) const {
  std::shared_lock<std::shared_mutex> read_lock(mutex_);

  try {
    nlohmann::json json_data;
    json_data["ModelPath"] = model_path_;
    json_data["DeviceName"] = device_name_;
    json_data["DeviceLocation"] = device_location_;
    json_data["ServerApiKey"] = server_api_key_;
    json_data["ServerAPI"] = server_api_;
    json_data["ServerEntryEndpoint"] = server_entry_endpoint_;
    json_data["ServerExitEndpoint"] = server_exit_endpoint_;

    // Save entry and exit zones.
    for (const auto& zone : typed_zones_) {
      std::vector<int> zone_coords = {zone.zone.x1, zone.zone.y1, zone.zone.x2,
                                      zone.zone.y2};
      if (zone.type == tracker::ZoneType::ENTRY) {
        json_data["entry_box"] = zone_coords;
      } else if (zone.type == tracker::ZoneType::EXIT) {
        json_data["exit_box"] = zone_coords;
      }
    }

    for (const auto& [key, value] : config_data_) {
      try {
        if (std::holds_alternative<std::string>(value)) {
          json_data[key] = std::get<std::string>(value);
        } else if (std::holds_alternative<int>(value)) {
          json_data[key] = std::get<int>(value);
        } else if (std::holds_alternative<double>(value)) {
          json_data[key] = std::get<double>(value);
        } else if (std::holds_alternative<bool>(value)) {
          json_data[key] = std::get<bool>(value);
        }
      } catch (const std::bad_variant_access& e) {
        ERROR("Failed to convert config value for key " << key << ": " << e.what());
      }
    }

    std::ofstream file(file_path);
    if (!file.is_open()) {
      ERROR("Failed to open file for saving configuration: " << file_path);
      return false;
    }

    file << json_data.dump(4);
    file.close();
    LOG("Configuration saved to: " << file_path);
    return true;
  } catch (const std::exception& e) {
    ERROR("Error saving configuration: " << e.what());
    return false;
  }
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

void Configuration::LoadConfigFile(const std::string& config_file_path) {
  std::unique_lock<std::shared_mutex> write_lock(mutex_);

  std::ifstream file(config_file_path);
  if (!file.is_open()) {
    ERROR("Failed to open config file: " << config_file_path);
    throw std::runtime_error("Config file not found: " + config_file_path);
  }

  nlohmann::json json_data;
  try {
    file >> json_data;
  } catch (const nlohmann::json::exception& e) {
    ERROR("Failed to parse config file " << config_file_path << ": " << e.what());
    throw std::runtime_error("Invalid JSON format in config file: " +
                             std::string(e.what()));
  }

  model_path_ = json_data.value("ModelPath", std::string());
  device_name_ = json_data.value("DeviceName", std::string());
  device_location_ = json_data.value("DeviceLocation", std::string());
  server_api_key_ = json_data.value("ServerApiKey", std::string());
  server_api_ = json_data.value("ServerAPI", std::string());
  server_entry_endpoint_ = json_data.value("ServerEntryEndpoint", std::string());
  server_exit_endpoint_ = json_data.value("ServerExitEndpoint", std::string());

  // Process entry and exit zones.
  typed_zones_.clear();
  if (json_data.contains("entry_box")) {
    if (!json_data["entry_box"].is_array() ||
        json_data["entry_box"].size() != 4) {
      ERROR("Invalid entry_box format in file: Expected array of 4 integers [x1, y1, x2, y2]");
    } else {
      try {
        auto zone_array = json_data["entry_box"].get<std::vector<int>>();
        tracker::BoxZone zone = {zone_array[0], zone_array[1], zone_array[2],
                                 zone_array[3]};
        if (zone.x1 < zone.x2 && zone.y1 < zone.y2) {
          typed_zones_.emplace_back(zone, tracker::ZoneType::ENTRY);
          LOG("Loaded entry zone: (x1=" << zone.x1 << ", y1=" << zone.y1
                                       << ", x2=" << zone.x2 << ", y2=" << zone.y2
                                       << ")");
        } else {
          ERROR("Invalid entry_box coordinates in file: x1 >= x2 or y1 >= y2");
        }
      } catch (const nlohmann::json::exception& e) {
        ERROR("Failed to parse entry_box from file: " << e.what());
      }
    }
  }

  if (json_data.contains("exit_box")) {
    if (!json_data["exit_box"].is_array() ||
        json_data["exit_box"].size() != 4) {
      ERROR("Invalid exit_box format in file: Expected array of 4 integers [x1, y1, x2, y2]");
    } else {
      try {
        auto zone_array = json_data["exit_box"].get<std::vector<int>>();
        tracker::BoxZone zone = {zone_array[0], zone_array[1], zone_array[2],
                                 zone_array[3]};
        if (zone.x1 < zone.x2 && zone.y1 < zone.y2) {
          typed_zones_.emplace_back(zone, tracker::ZoneType::EXIT);
          LOG("Loaded exit zone: (x1=" << zone.x1 << ", y1=" << zone.y1
                                      << ", x2=" << zone.x2 << ", y2=" << zone.y2
                                      << ")");
        } else {
          ERROR("Invalid exit_box coordinates in file: x1 >= x2 or y1 >= y2");
        }
      } catch (const nlohmann::json::exception& e) {
        ERROR("Failed to parse exit_box from file: " << e.what());
      }
    }
  }

  // Load additional custom settings.
  for (auto it = json_data.begin(); it != json_data.end(); ++it) {
    const std::string& key = it.key();
    if (key == "ModelPath" || key == "DeviceName" || key == "DeviceLocation" ||
        key == "ServerApiKey" || key == "ServerAPI" ||
        key == "ServerEntryEndpoint" || key == "ServerExitEndpoint" ||
        key == "entry_box" || key == "exit_box") {
      continue;
    }

    if (it->is_string()) {
      config_data_[key] = it->get<std::string>();
    } else if (it->is_number_integer()) {
      config_data_[key] = it->get<int>();
    } else if (it->is_number_float()) {
      config_data_[key] = it->get<double>();
    } else if (it->is_boolean()) {
      config_data_[key] = it->get<bool>();
    }
  }

  LOG("Configuration loaded from file: " << config_file_path << " with "
                                         << typed_zones_.size() << " zones");
}