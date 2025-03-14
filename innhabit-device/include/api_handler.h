#ifndef API_HANDLER_H
#define API_HANDLER_H

#include <curl/curl.h>
#include <string>
#include <nlohmann/json.hpp>
#include <mutex>

class TrackedPerson;  // Forward declaration

class ApiHandler {
private:
    CURL* curl;
    std::mutex curlMutex;
    std::string baseUrl;
    
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data);
    void Initialize();
    // Private method returning only datetime string
    std::string getTimestamp();

public:
    ApiHandler(const std::string& url = "http://localhost:8000");
    ~ApiHandler();

    nlohmann::json sendPostRequest(const std::string& endpoint, const nlohmann::json& data);
    bool onPersonEvent(const TrackedPerson& person, const std::string& eventType);
    void saveResponseToFile(const nlohmann::json& response, const std::string& filename);
};

#endif // API_HANDLER_H