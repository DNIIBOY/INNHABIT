#include "api_handler.h"
#include "tracker.h"  // For TrackedPerson definition
#include <iostream>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <ctime>  // For std::strftime

// Namespaces
using namespace std;
using json = nlohmann::json;

ApiHandler::ApiHandler(const string& url) : baseUrl(url) {
    Initialize();
}

ApiHandler::~ApiHandler() {
    if (curl) {
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
}

void ApiHandler::Initialize() {
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (!curl) {
        throw runtime_error("CURL initialization failed");
    }
}

size_t ApiHandler::WriteCallback(void* contents, size_t size, size_t nmemb, string* data) {
    size_t totalSize = size * nmemb;
    data->append((char*)contents, totalSize);
    return totalSize;
}

json ApiHandler::sendPostRequest(const string& endpoint, const json& data) {
    lock_guard<mutex> lock(curlMutex);
    
    string response_string;
    string url = baseUrl + endpoint;
    string json_str = data.dump();

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());
    
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

    CURLcode res = curl_easy_perform(curl);
    
    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        throw runtime_error("curl_easy_perform() failed: " + 
                          string(curl_easy_strerror(res)));
    }

    return json::parse(response_string);
}

string ApiHandler::getTimestamp() {
    
    // Convert to seconds for readable format
    auto current_time = chrono::duration_cast<chrono::seconds>(
        chrono::system_clock::now().time_since_epoch()).count();
    
    // Convert to time_t for formatting
    time_t tt = static_cast<time_t>(current_time);
    
    // Format to human-readable string
    char buffer[80];
    struct tm* timeinfo = gmtime(&tt);  // UTC time
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    
    // Add milliseconds
    return string(buffer) + " UTC";
}

bool ApiHandler::onPersonEvent(const TrackedPerson& person, const std::string& eventType) {
    try {
        string timestamp_str = getTimestamp();
        json request_data = {
            {"datetime", timestamp_str},
            {"event", eventType}
        };
        
        json response = sendPostRequest("/PersonEvent", request_data);
        if (response["status"] == "success") {
            return true;
        }
        return false;
    } catch (const exception& e) {
        cerr << "Error sending entry event: " << e.what() << endl;
        return false;
    }
}

void ApiHandler::saveResponseToFile(const json& response, const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        file << response.dump(4);
        file.close();
    }
}