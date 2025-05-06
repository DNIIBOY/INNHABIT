#include "devicePoll.h"
#include <iostream>
#include <chrono>

using json = nlohmann::json;

DevicePoller::DevicePoller(const std::string& apiUrl, 
    const std::string& apiKey, 
    std::atomic<bool>& shouldExit, 
    std::queue<cv::Mat>& frameQueue, 
    std::mutex& frameMutex, 
    std::condition_variable& frameCV, 
    ApiHandler* apiHandler,
    std::shared_ptr<Configuration> config,
    int pollIntervalSeconds)
: m_apiUrl(apiUrl), m_apiKey(apiKey), m_shouldExit(shouldExit),
  m_frameQueue(frameQueue), m_frameMutex(frameMutex), m_frameCV(frameCV),
  m_apiHandler(apiHandler), m_config(config), m_pollIntervalSeconds(pollIntervalSeconds),
  m_curl(nullptr) {
    curl_global_init(CURL_GLOBAL_ALL);
    m_curl = curl_easy_init();
    if (!m_curl) {
        throw std::runtime_error("CURL initialization failed in DevicePoller");
    }
}

DevicePoller::~DevicePoller() {
    if (m_curl) {
        curl_easy_cleanup(m_curl);
    }
    curl_global_cleanup();
}

void DevicePoller::start() {
    m_thread = std::thread(&DevicePoller::pollServer, this);
}

void DevicePoller::join() {
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void DevicePoller::pollServer() {
    while (!m_shouldExit) {
        try {
            json response = sendGetRequest("device/poll/");
            if (!response.is_null() && !response.empty() && response.is_array()) {
                LOG("Poll response: " << response.dump());

                // Process each action in the array
                for (const auto& item : response) {
                    if (!item.is_string()) {
                        LOG("Skipping non-string item in response array: " << item.dump());
                        continue; // Skip invalid items
                    }

                    std::string action = item.get<std::string>();
                    LOG("Processing action: " << action);

                    if (action == "request_image") {
                        LOG("Received request_image command from server");
                        if (captureAndSendImage()) {
                            LOG("Image sent successfully");
                        } else {
                            ERROR("Failed to send image");
                        }
                    } else if (action == "update_settings") {
                        LOG("Received update_settings command from server");
                        json settings = sendGetRequest("device/settings/");
                        if (!settings.is_null() && !settings.empty()) {
                            m_config->UpdateFromJson(settings);
                            m_config->SaveToFile("../settings.json");
                            LOG("Settings saved to file");
                            LOG("Settings updated successfully: " << settings.dump());
                        } else {
                            ERROR("Failed to fetch new settings from API");
                        }
                    } else {
                        LOG("Unknown action: " << action);
                    }
                }
            } else {
                LOG("No actions to process (empty, null, or not an array): " 
                    << (response.is_null() ? "null" : response.dump()));
            }
        } catch (const std::exception& e) {
            ERROR("Polling error: " << e.what());
        }

        std::this_thread::sleep_for(std::chrono::seconds(m_pollIntervalSeconds));
    }
}

bool DevicePoller::captureAndSendImage() {
    if (!m_apiHandler) {
        ERROR("API handler not initialized");
        return false;
    }

    cv::Mat frame;
    {
        std::unique_lock<std::mutex> lock(m_frameMutex);
        m_frameCV.wait_for(lock, std::chrono::milliseconds(100), [this] { 
            return !m_frameQueue.empty() || m_shouldExit; 
        });

        if (m_shouldExit) {
            return false;
        }

        if (!m_frameQueue.empty()) {
            frame = m_frameQueue.front();
            m_frameQueue.pop();
        } else {
            ERROR("No frame available in frameQueue");
            return false;
        }
    }

    if (frame.empty()) {
        ERROR("Captured frame is empty");
        return false;
    }

    return m_apiHandler->sendImage(frame);
}

json DevicePoller::sendGetRequest(const std::string& endpoint) {
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("Authorization: Bearer " + m_apiKey).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "User-Agent: InnhabitDevice/1.0");

    std::string fullUrl = m_apiUrl + endpoint;
    curl_easy_setopt(m_curl, CURLOPT_URL, fullUrl.c_str());
    curl_easy_setopt(m_curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(m_curl, CURLOPT_HTTPGET, 1L);

    std::string response;
    curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(m_curl, CURLOPT_WRITEDATA, &response);

    long http_code = 0;
    CURLcode res = curl_easy_perform(m_curl);
    curl_easy_getinfo(m_curl, CURLINFO_RESPONSE_CODE, &http_code);

    LOG("HTTP Status Code: " << http_code << ", Raw response: " << (response.empty() ? "<empty>" : response));

    json result;
    if (res == CURLE_OK) {
        if (http_code == 200) {
            try {
                result = json::parse(response, nullptr, false);
                if (result.is_discarded()) {
                    ERROR("Failed to parse poll response as JSON");
                    result = json();
                }
            } catch (const json::parse_error& e) {
                ERROR("Parse error in poll response: " << e.what());
                result = json();
            }
        } else if (http_code == 204) {
            // No content, return empty JSON to indicate no actions
            result = json();
        } else {
            ERROR("Unexpected HTTP Code: " << http_code);
            result = json();
        }
    } else {
        ERROR("CURL error: " << curl_easy_strerror(res));
        result = json();
    }

    curl_slist_free_all(headers);
    return result;
}

size_t DevicePoller::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t totalSize = size * nmemb;
    data->append((char*)contents, totalSize);
    return totalSize;
}
