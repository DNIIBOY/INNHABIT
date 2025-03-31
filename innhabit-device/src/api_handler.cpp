#include "api_handler.h"
#include "tracker.h"  // For TrackedPerson definition
#include <iostream>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <ctime>  // For std::strftime
#include <opencv2/opencv.hpp> // For image handling

ApiHandler::ApiHandler(const std::string& url, const std::string& api_key) 
        : base_url_(url),
        api_key_(api_key), 
        curl_(nullptr), 
        should_exit_(false) {
    initialize();
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
        if (has_event) {
        processSingleEvent(event);
        }
    }
    
    // Process any remaining events in the queue before exiting
    processRemainingEvents();
}

void ApiHandler::processSingleEvent(const ApiEvent& event) {
    try {
        json request_data = {
        {"timestamp", event.timestamp},
        {"entrance", 1} // Adjust based on actual logic (e.g., zone ID)
        };
        
        json response;
        if (event.event_type == "entered") {
        response = sendPostRequest("events/exits/", request_data); // Note: Seems reversed?
        } else if (event.event_type == "exited") {
        response = sendPostRequest("events/entries/", request_data);
        } else {
        ERROR("ApiHandler: Unknown event type: " << event.event_type);
        return;
        }

        if (response.is_null() || response.empty()) {
        ERROR("Empty or null response from server");
        } else {
        LOG("API response success: " << response.dump());
        }
    } catch (const std::exception& e) {
        ERROR("Error sending event to server: " << e.what());
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
        
        if (event.event_type == "entered") {
            sendPostRequest("events/exits/", request_data);
        } else if (event.event_type == "exited") {
            sendPostRequest("events/entries/", request_data);
        }
        } catch (...) {
        // Just log the error and continue processing remaining events
        ERROR("Error processing final event");
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

    // Add Authorization header with API key
    std::string auth_header = "Authorization: Bearer " + api_key_;
    headers = curl_slist_append(headers, auth_header.c_str());

    // Construct full URL
    std::string full_url = base_url_ + endpoint;
    curl_easy_setopt(curl_, CURLOPT_URL, full_url.c_str());

    // Convert JSON to string for POST data
    std::string data_str = data.dump();
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, data_str.c_str());

    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    std::string response;
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);

    // Get HTTP status code
    long http_code = 0;
    CURLcode res = curl_easy_perform(curl_);
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);

    json result;
    if (res == CURLE_OK) {
        if (http_code >= 200 && http_code < 300) { // Success range
        try {
            result = json::parse(response, nullptr, false);
            if (result.is_discarded()) {
            ERROR("Failed to parse response as JSON");
            result = json(); // Return empty JSON on parse failure
            }
        } catch (const json::parse_error& e) {
            ERROR("Parse error: " << e.what());
            result = json(); // Return empty JSON on exception
        }
        } else {
        ERROR("HTTP error: " << http_code << " - Response: " << response);
        result = json(); // Return empty JSON on HTTP error
        }
    } else {
        ERROR("CURL error: " << curl_easy_strerror(res));
        result = json(); // Return empty JSON on CURL failure
    }

    curl_slist_free_all(headers);
    return result; // Always return a json object
}

std::string ApiHandler::getTimestampISO() {
    auto now = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(now);
    char buffer[80];
    struct tm* timeinfo = gmtime(&tt);  // UTC time
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", timeinfo);
    return std::string(buffer);
}

bool ApiHandler::onPersonEvent(const std::string& event_type) {
    try {
        // Create event and add to queue
        ApiEvent event;
        event.event_type = event_type;
        event.timestamp = getTimestampISO();
        
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
        // Convert cv::Mat to a temporary file (e.g., PNG format)
        std::string temp_filename = "temp_image_" + getTimestampISO() + ".png";
        cv::imwrite(temp_filename, image);

        // Prepare multipart form-data
        curl_mime* mime = curl_mime_init(curl_);
        curl_mimepart* part = curl_mime_addpart(mime);

        // Add the image file to the form-data
        curl_mime_name(part, "image");
        curl_mime_filedata(part, temp_filename.c_str());
        curl_mime_type(part, "image/png");

        // Add Authorization header
        struct curl_slist* headers = nullptr;
        std::string auth_header = "Authorization: Bearer " + api_key_;
        headers = curl_slist_append(headers, auth_header.c_str());

        // Set up CURL options
        std::string url = base_url_ + "images/";
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl_, CURLOPT_MIMEPOST, mime);

        std::string response;
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);

        // Perform the request
        CURLcode res = curl_easy_perform(curl_);
        long http_code = 0;
        curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);

        // Clean up
        curl_mime_free(mime);
        curl_slist_free_all(headers);
        std::remove(temp_filename.c_str()); // Delete temporary file

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

void ApiHandler::saveResponseToFile(const json& response, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << response.dump(4); // Pretty print with indentation
        file.close();
    }
}