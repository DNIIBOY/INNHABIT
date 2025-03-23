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

ApiHandler::ApiHandler(const string& url) 
    : baseUrl(url), 
      curl(nullptr), 
      m_shouldExit(false) {
    Initialize();
}

ApiHandler::~ApiHandler() {
    shutdown();
    join();
    
    if (curl) {
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
}

void ApiHandler::start() {
    m_thread = std::thread(&ApiHandler::processEvents, this);
}

void ApiHandler::shutdown() {
    m_shouldExit = true;
    m_queueCV.notify_one();
}

void ApiHandler::join() {
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void ApiHandler::Initialize() {
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (!curl) {
        throw runtime_error("CURL initialization failed");
    }
}

void ApiHandler::processEvents() {
    while (!m_shouldExit) {
        ApiEvent event;
        bool hasEvent = false;
        
        // Get event from queue
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCV.wait(lock, [this] { 
                return !m_eventQueue.empty() || m_shouldExit; 
            });
            
            if (m_shouldExit && m_eventQueue.empty()) {
                break;
            }
            
            if (!m_eventQueue.empty()) {
                event = m_eventQueue.front();
                m_eventQueue.pop();
                hasEvent = true;
            }
        }
        
        // Process the event
        if (hasEvent) {
            try {
                json request_data = {
                    {"timestamp", event.timestamp},
                    {"entrance", 1} // Adjust based on actual logic (e.g., zone ID)
                };
                
                json response;
                if (event.eventType == "entered") {
                    response = sendPostRequest("/exits/", request_data); // Note: Seems reversed?
                } else if (event.eventType == "exited") {
                    response = sendPostRequest("/entries/", request_data);
                } else {
                    ERROR("Apihandler: Unknown event type: " << event.eventType);
                    continue;
                }

                if (response.is_null() || response.empty()) {
                    ERROR("Empty or null response from server");
                } else {
                    cout << "API response success: " << response.dump() << endl;
                }
            } catch (const exception& e) {
                cerr << "Error sending event to server: " << e.what() << endl;
            }
        }
    }
    
    // Process any remaining events in the queue before exiting
    std::unique_lock<std::mutex> lock(m_queueMutex);
    while (!m_eventQueue.empty()) {
        ApiEvent event = m_eventQueue.front();
        m_eventQueue.pop();
        lock.unlock();
        
        // Similar processing as above, but simplified error handling
        try {
            json request_data = {
                {"timestamp", event.timestamp},
                {"entrance", 1}
            };
            
            if (event.eventType == "entered") {
                sendPostRequest("/exits/", request_data);
            } else if (event.eventType == "exited") {
                sendPostRequest("/entries/", request_data);
            }
        } catch (...) {
            // Just log the error and continue processing remaining events
            cerr << "Error processing final event" << endl;
        }
        
        lock.lock();
    }
}

size_t ApiHandler::WriteCallback(void* contents, size_t size, size_t nmemb, string* data) {
    size_t totalSize = size * nmemb;
    data->append((char*)contents, totalSize);
    return totalSize;
}

json ApiHandler::sendPostRequest(const string& endpoint, const json& data) {
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    // Construct full URL
    string fullUrl = baseUrl + endpoint;
    curl_easy_setopt(curl, CURLOPT_URL, fullUrl.c_str());

    // Convert JSON to string for POST data
    string dataStr = data.dump();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, dataStr.c_str());

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    string response;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback); // Use correct name
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    // Get HTTP status code
    long http_code = 0;
    CURLcode res = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    json result;
    if (res == CURLE_OK) {
        if (http_code >= 200 && http_code < 300) { // Success range
            try {
                result = json::parse(response, nullptr, false);
                if (result.is_discarded()) {
                    cerr << "Failed to parse response as JSON" << endl;
                    result = json(); // Return empty JSON on parse failure
                }
            } catch (const json::parse_error& e) {
                cerr << "Parse error: " << e.what() << endl;
                result = json(); // Return empty JSON on exception
            }
        } else {
            cerr << "HTTP error: " << http_code << " - Response: " << response << endl;
            result = json(); // Return empty JSON on HTTP error
        }
    } else {
        cerr << "CURL error: " << curl_easy_strerror(res) << endl;
        result = json(); // Return empty JSON on CURL failure
    }

    curl_slist_free_all(headers);
    return result; // Always return a json object
}

string ApiHandler::getTimestamp() {
    auto current_time = chrono::duration_cast<chrono::seconds>(
        chrono::system_clock::now().time_since_epoch()).count();
    time_t tt = static_cast<time_t>(current_time);
    char buffer[80];
    struct tm* timeinfo = gmtime(&tt);  // UTC time
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    return string(buffer) + " UTC";
}

string ApiHandler::getTimestampISO() {
    auto now = chrono::system_clock::now();
    time_t tt = chrono::system_clock::to_time_t(now);
    char buffer[80];
    struct tm* timeinfo = gmtime(&tt);  // UTC time
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", timeinfo);
    return string(buffer);
}

bool ApiHandler::onPersonEvent(const std::string& eventType) {
    try {
        // Create event and add to queue
        ApiEvent event;
        event.eventType = eventType;
        event.timestamp = getTimestampISO();
        
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_eventQueue.push(event);
        }
        
        m_queueCV.notify_one();
        return true;
    } catch (const exception& e) {
        cerr << "Error queueing event: " << e.what() << endl;
        return false;
    }
}

void ApiHandler::saveResponseToFile(const json& response, const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        file << response.dump(4); // Pretty print with indentation
        file.close();
    }
}