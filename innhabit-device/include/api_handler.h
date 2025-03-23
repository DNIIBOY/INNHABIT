#ifndef API_HANDLER_H
#define API_HANDLER_H

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

struct ApiEvent {
    std::string eventType;
    std::string timestamp;
};

class ApiHandler {
public:
    ApiHandler(const std::string& url);
    ~ApiHandler();

    // Non-blocking call to submit an event
    bool onPersonEvent(const std::string& eventType);
    
    // Start the API handler thread
    void start();
    
    // Shutdown the API handler thread
    void shutdown();
    
    // Wait for the API thread to complete
    void join();

    void setBaseUrl(const std::string& url) { baseUrl = url;}

private:
    // Thread function
    void processEvents();
    
    // Initialize CURL
    void Initialize();
    
    // HTTP helpers
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data);
    nlohmann::json sendPostRequest(const std::string& endpoint, const nlohmann::json& data);
    
    // Time helpers
    std::string getTimestamp();
    std::string getTimestampISO();
    
    // Debug helper
    void saveResponseToFile(const nlohmann::json& response, const std::string& filename);

    // CURL handle
    CURL* curl;
    
    // Server URL
    std::string baseUrl;
    
    // Thread management
    std::thread m_thread;
    std::atomic<bool> m_shouldExit;
    
    // Event queue
    std::queue<ApiEvent> m_eventQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;
};

#endif // API_HANDLER_H