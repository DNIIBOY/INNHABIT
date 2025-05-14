#ifndef API_HANDLER_H
#define API_HANDLER_H

#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <vector>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "common.h"

struct ApiEvent {
    std::string event_type;
    std::string timestamp;
};

class ApiHandler {
public:
    ApiHandler(std::shared_ptr<Configuration> config);
    ~ApiHandler();
    void start();
    void shutdown();
    void join();
    bool onPersonEvent(const std::string& event_type);
    bool sendImage(const cv::Mat& image);
    void saveResponseToFile(const nlohmann::json& response, const std::string& filename);

private:
    using Json = nlohmann::json;
    void initialize();
    void processEvents();
    void processSingleEvent(const ApiEvent& event);
    void processRemainingEvents();
    Json sendPostRequest(const std::string& endpoint, const Json& data);
    static size_t writeCallback(void* contents, size_t size, size_t nmemb, std::string* data);
    std::string getTimestampISO();
    void saveFailedEventsToDisk(const ApiEvent api_event);
    void loadFailedEventsFromDisk();
    
    std::string api_key_;
    std::string base_url_;
    CURL* curl_;
    bool should_exit_;
    bool failed_event_;
    bool last_api_success_;
    std::thread thread_;
    std::queue<ApiEvent> event_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<ApiEvent> failed_events_;
};

#endif // API_HANDLER_H