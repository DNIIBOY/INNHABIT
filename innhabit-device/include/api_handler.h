#ifndef API_HANDLER_H
#define API_HANDLER_H

#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "common.h" // Include for logging macros

// Define the ApiEvent structure
struct ApiEvent {
    std::string event_type;
    std::string timestamp;
};

class ApiHandler {
public:
    ApiHandler(std::shared_ptr<Configuration> config);
    ~ApiHandler();

    // Thread management
    /**
    * Start the API handler on its own thread
    */
    void start();
    void shutdown();
    /**
    * Joins the API handler thread for safe exiting
    */
    void join();
    
    // Event handling
    bool onPersonEvent(const std::string& event_type);
    bool sendImage(const cv::Mat& image);

    // Utility functions
    void SaveResponseToFile(const nlohmann::json& response, const std::string& filename);
    
private:
    using Json = nlohmann::json;
    // Initialization and processing
    void initialize();
    void processEvents();
    void processSingleEvent(const ApiEvent& event);
    void processRemainingEvents();
    // HTTP request handling
    nlohmann::json sendPostRequest(const std::string& endpoint, const nlohmann::json& data);
    static size_t writeCallback(void* contents, size_t size, size_t nmemb, std::string* data);
    
    // Time utilities
    std::string getTimestampISO();
    
    // backup 
    void saveResponseToFile(const nlohmann::json& response, const std::string& filename);

    // Member variables
    std::string api_key_;
    std::string base_url_;
    CURL* curl_;
    bool should_exit_;
    
    std::thread thread_;
    std::queue<ApiEvent> event_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
};

#endif // API_HANDLER_H
