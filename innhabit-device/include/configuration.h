#ifndef INCLUDE_CONFIGURATION_H_
#define INCLUDE_CONFIGURATION_H_

#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "common.h"
#include "tracker/zone_type.h"

// Thread-safe singleton class for managing application settings.
class Configuration : public std::enable_shared_from_this<Configuration> {
 public:
  // Gets the singleton instance of the Configuration class.
  static std::shared_ptr<Configuration> GetInstance();

  // Loads configuration settings from a JSON file.
  // Args:
  //   config_file_path: Path to the configuration JSON file.
  // Returns:
  //   A shared pointer to the Configuration object.
  // Throws:
  //   std::runtime_error if the file cannot be opened or JSON parsing fails.
  static std::shared_ptr<Configuration> LoadFromFile(
      const std::string& config_file_path);

  void UpdateFromApi(const nlohmann::json& json_data);

  // Updates configuration settings from a JSON object.
  // Args:
  //   json_data: JSON object containing new settings.
  void UpdateFromJson(const nlohmann::json& json_data);

  // Saves the current configuration to a JSON file.
  // Args:
  //   file_path: Path where the configuration file should be saved.
  // Returns:
  //   True if saving was successful, false otherwise.
  bool SaveToFile(const std::string& file_path) const;

  // Gets the path to the ML model.
  std::string GetModelPath() const;

  // Gets the name of the device.
  std::string GetDeviceName() const;

  // Gets the location of the device.
  std::string GetDeviceLocation() const;

  // Gets the API key for server communication.
  std::string GetServerApiKey() const;

  // Gets the server API base URL.
  std::string GetServerApi() const;

  // Gets the endpoint for entry events.
  std::string GetServerEntryEndpoint() const;

  // Gets the endpoint for exit events.
  std::string GetServerExitEndpoint() const;

  // Gets the list of typed zones (entry and exit).
  std::vector<tracker::TypedZone> GetTypedZones() const;

  // Gets a configuration value for a given key with type conversion.
  // Args:
  //   key: The configuration key to retrieve.
  // Returns:
  //   An optional containing the value if found and convertible to T, else empty.
  template <typename T>
  std::optional<T> GetValue(const std::string& key) const;

  // Sets a configuration value for a given key.
  // Args:
  //   key: The configuration key to set.
  //   value: The value to set.
  template <typename T>
  void SetValue(const std::string& key, const T& value);

  // Checks if a configuration key exists.
  // Args:
  //   key: The configuration key to check.
  // Returns:
  //   True if the key exists, false otherwise.
  bool HasKey(const std::string& key) const;

  // Removes a configuration key.
  // Args:
  //   key: The configuration key to remove.
  // Returns:
  //   True if the key was removed, false if it didn't exist.
  bool RemoveKey(const std::string& key);

 private:
  Configuration() = default;

  // Delete copy and move operations to enforce singleton pattern.
  Configuration(const Configuration&) = delete;
  Configuration& operator=(const Configuration&) = delete;
  Configuration(Configuration&&) = delete;
  Configuration& operator=(Configuration&&) = delete;

  // Loads configuration from a JSON file.
  // Args:
  //   config_file_path: Path to the JSON file.
  // Throws:
  //   std::runtime_error if the file cannot be opened or JSON parsing fails.
  void LoadConfigFile(const std::string& config_file_path);

  mutable std::shared_mutex mutex_;  // Read-write lock for thread safety.
  std::string model_path_;           // Path to the ML model.
  std::string device_name_;          // Name of the device.
  std::string device_location_;      // Location of the device.
  std::string server_api_key_;       // API key for server communication.
  std::string server_api_;           // Server API base URL.
  std::string server_entry_endpoint_;  // Endpoint for entry events.
  std::string server_exit_endpoint_;   // Endpoint for exit events.
  std::vector<tracker::TypedZone> typed_zones_;  // List of entry and exit zones.
  std::unordered_map<std::string, std::variant<std::string, int, double, bool>>
      config_data_;  // Custom configuration settings.
};

template <typename T>
std::optional<T> Configuration::GetValue(const std::string& key) const {
  std::shared_lock<std::shared_mutex> read_lock(mutex_);

  auto it = config_data_.find(key);
  if (it != config_data_.end()) {
    try {
      return std::get<T>(it->second);
    } catch (const std::bad_variant_access&) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

template <typename T>
void Configuration::SetValue(const std::string& key, const T& value) {
  std::unique_lock<std::shared_mutex> write_lock(mutex_);
  config_data_[key] = value;
}

#endif  // INCLUDE_CONFIGURATION_H_