#ifndef API_HANDLER_H
#define API_HANDLER_H

#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

// Using just the symbols we need rather than entire namespaces
using json = nlohmann::json;
// Define the ApiEvent structure
struct ApiEvent {
    std::string event_type;
    std::string timestamp;
};

class ApiHandler {
public:
    ApiHandler(const std::string& url, const std::string& api_key);
    ~ApiHandler();

    // Thread management
    void start();
    void shutdown();
    void join();
    
    // Event handling
    bool onPersonEvent(const std::string& event_type);
    
    // Utility functions
    void SaveResponseToFile(const nlohmann::json& response, const std::string& filename);

private:
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
    void saveResponseToFile(const json& response, const std::string& filename);

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