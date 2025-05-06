#include "api_handler.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <ctime>  // For std::strftime

ApiHandler::ApiHandler(std::shared_ptr<Configuration> config) 
    : base_url_(config->getServerApi()), api_key_(config->getServerApiKey()), 
      curl_(nullptr), should_exit_(false), failed_event_(false) {
    initialize();
    loadFailedEventsFromDisk();
}

ApiHandler::~ApiHandler() {
    shutdown();
    join();
    
    if (curl_) {
        curl_easy_cleanup(curl_);
    }
    curl_global_cleanup();
}

void ApiHandler::start() {
    thread_ = std::thread(&ApiHandler::processEvents, this);
}

void ApiHandler::shutdown() {
    should_exit_ = true;
    queue_cv_.notify_one();
}

void ApiHandler::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void ApiHandler::initialize() {
    curl_global_init(CURL_GLOBAL_ALL);
    curl_ = curl_easy_init();
    if (!curl_) {
        throw std::runtime_error("CURL initialization failed");
    }
}

void ApiHandler::processEvents() {
    while (!should_exit_) {
        ApiEvent event;
        bool has_event = false;
        
        // Get event from queue
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { 
                return !event_queue_.empty() || should_exit_; 
            });
            
            if (should_exit_ && event_queue_.empty()) {
                break;
            }
            
            if (!event_queue_.empty()) {
                event = event_queue_.front();
                event_queue_.pop();
                has_event = true;
            }
        }
        
        // Process the event
        if (has_event && !should_exit_) {
            processSingleEvent(event);
        }
    }
    
    // Process any remaining events in the queue before exiting
    processRemainingEvents();
}

void ApiHandler::processSingleEvent(const ApiEvent& event) {
    try {
        json request_data = {
            {"timestamp", event.timestamp}
        };
        
        json response;
        if (event.event_type == "exited") {
            response = sendPostRequest("events/exits/", request_data);
        } else if (event.event_type == "entered") {
            response = sendPostRequest("events/entries/", request_data);
        } else if (event.event_type == "exited") {
            response = sendPostRequest("events/exits/", request_data);
        } else {
            ERROR("ApiHandler: Unknown event type: " << event.event_type);
            return;
        }

        if (response.is_null() || response.empty()) {
            ERROR("Empty or null response from server");
            saveFailedEventsToDisk(event);  // Save on failure
        }
    } catch (const std::exception& e) {
        ERROR("Error sending event to server: " << e.what());
        saveFailedEventsToDisk(event);  // Save on exception
    }
}

void ApiHandler::processRemainingEvents() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    while (!event_queue_.empty()) {
        ApiEvent event = event_queue_.front();
        event_queue_.pop();
        lock.unlock();
        
        try {
            json request_data = {
                {"timestamp", event.timestamp},
                {"entrance", 1}
            };
            
            if (event.event_type == "exited") {
                sendPostRequest("events/exits/", request_data);
            } else if (event.event_type == "entered") {
                sendPostRequest("events/entries/", request_data);
            }
        } catch (...) {
            ERROR("Error processing final event");
            saveFailedEventsToDisk(event);
            failed_event_ = true;
        }
        
        lock.lock();
    }
}

size_t ApiHandler::writeCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t total_size = size * nmemb;
    data->append(static_cast<char*>(contents), total_size);
    return total_size;
}

json ApiHandler::sendPostRequest(const std::string& endpoint, const json& data) {
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    std::string auth_header = "Authorization: Bearer " + api_key_;
    headers = curl_slist_append(headers, auth_header.c_str());

    std::string full_url = base_url_ + endpoint;
    curl_easy_setopt(curl_, CURLOPT_URL, full_url.c_str());

    std::string data_str = data.dump();
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, data_str.c_str());

    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    std::string response;
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);

    long http_code = 0;
    CURLcode res = curl_easy_perform(curl_);
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);

    json result;
    if (res == CURLE_OK) {
        if (http_code >= 200 && http_code < 300) {
            try {
                result = json::parse(response, nullptr, false);
                if (result.is_discarded()) {
                    ERROR("Failed to parse response as JSON");
                    result = json();
                }
            } catch (const json::parse_error& e) {
                ERROR("Parse error: " << e.what());
                result = json();
            }
        } else {
            ERROR("HTTP error: " << http_code << " - Response: " << response);
            result = json();
        }
    } else {
        ERROR("CURL error: " << curl_easy_strerror(res));
        result = json();
    }

    curl_slist_free_all(headers);
    return result;
}

std::string ApiHandler::getTimestampISO() {
    auto now = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(now);
    char buffer[80];
    struct tm* timeinfo = gmtime(&tt);
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", timeinfo);
    return std::string(buffer);
}

bool ApiHandler::onPersonEvent(const std::string& event_type) {
    try {
        ApiEvent event;
        event.event_type = event_type;
        event.timestamp = getTimestampISO();
        LOG("Event: " << event.event_type << " at " << event.timestamp);
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            event_queue_.push(event);
        }
        
        queue_cv_.notify_one();
        return true;
    } catch (const std::exception& e) {
        ERROR("Error queueing event: " << e.what());
        return false;
    }
}

bool ApiHandler::sendImage(const cv::Mat& image) {
    if (!curl_) {
        ERROR("CURL not initialized");
        return false;
    }

    try {
        std::string temp_filename = "temp_image_" + getTimestampISO() + ".png";
        cv::imwrite(temp_filename, image);

        curl_mime* mime = curl_mime_init(curl_);
        curl_mimepart* part = curl_mime_addpart(mime);

        curl_mime_name(part, "image");
        curl_mime_filedata(part, temp_filename.c_str());
        curl_mime_type(part, "image/png");

        struct curl_slist* headers = nullptr;
        std::string auth_header = "Authorization: Bearer " + api_key_;
        headers = curl_slist_append(headers, auth_header.c_str());

        std::string url = base_url_ + "images/";
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl_, CURLOPT_MIMEPOST, mime);

        std::string response;
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);

        CURLcode res = curl_easy_perform(curl_);
        long http_code = 0;
        curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);

        curl_mime_free(mime);
        curl_slist_free_all(headers);
        std::remove(temp_filename.c_str());

        if (res != CURLE_OK) {
            ERROR("CURL error sending image: " << curl_easy_strerror(res));
            return false;
        }

        if (http_code >= 200 && http_code < 300) {
            LOG("Image uploaded successfully. Response: " << response);
            return true;
        } else {
            ERROR("HTTP error sending image: " << http_code << " - Response: " << response);
            return false;
        }
    } catch (const std::exception& e) {
        ERROR("Error sending image: " << e.what());
        return false;
    }
}

void ApiHandler::saveFailedEventsToDisk(const ApiEvent api_event) {
    try {
        failed_event_ = true;
        
        std::ofstream file("failed_events.bin", std::ios::binary | std::ios::app);
        if (!file.is_open()) {
            ERROR("Failed to open failed_events.bin for writing");
            return;
        }

        uint32_t event_type_len = static_cast<uint32_t>(api_event.event_type.length());
        file.write(reinterpret_cast<const char*>(&event_type_len), sizeof(event_type_len));
        file.write(api_event.event_type.c_str(), event_type_len);

        uint32_t timestamp_len = static_cast<uint32_t>(api_event.timestamp.length());
        file.write(reinterpret_cast<const char*>(&timestamp_len), sizeof(timestamp_len));
        file.write(api_event.timestamp.c_str(), timestamp_len);

        file.close();
        LOG("Successfully saved failed event to disk");
    } catch (const std::exception& e) {
        ERROR("Error saving failed event to disk: " << e.what());
    }
}

void ApiHandler::loadFailedEventsFromDisk() {
    try {
        std::ifstream file("failed_events.bin", std::ios::binary);
        if (!file.is_open()) {
            LOG("No failed_events.bin file found or unable to open");
            return;
        }

        while (file.good() && !file.eof()) {
            uint32_t event_type_len = 0;
            file.read(reinterpret_cast<char*>(&event_type_len), sizeof(event_type_len));
            if (file.eof()) break;

            std::string event_type;
            event_type.resize(event_type_len);
            file.read(&event_type[0], event_type_len);

            uint32_t timestamp_len = 0;
            file.read(reinterpret_cast<char*>(&timestamp_len), sizeof(timestamp_len));
            if (file.eof()) break;

            std::string timestamp;
            timestamp.resize(timestamp_len);
            file.read(&timestamp[0], timestamp_len);

            ApiEvent event{event_type, timestamp};
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                event_queue_.push(event);
            }
            LOG("Loaded failed event: " << event_type << " at " << timestamp);
        }

        file.close();
        queue_cv_.notify_one();

        if (std::remove("failed_events.bin") == 0) {
            LOG("Successfully removed failed_events.bin after loading");
        }
    } catch (const std::exception& e) {
        ERROR("Error loading failed events from disk: " << e.what());
    }
}
