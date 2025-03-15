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

bool ApiHandler::onPersonEvent(const TrackedPerson& person, const std::string& eventType) {
    try {
        string timestamp_str = getTimestampISO();
        json request_data = {
            {"timestamp", timestamp_str},
            {"entrance", 1} // Adjust based on actual logic (e.g., zone ID)
        };
        json response;
        if (eventType == "entered") {
            response = sendPostRequest("/exits/", request_data); // Note: Seems reversed?
        } else if (eventType == "exited") {
            response = sendPostRequest("/entries/", request_data);
        } else {
            cerr << "Unknown event type: " << eventType << endl;
            return false;
        }

        if (response.is_null() || response.empty()) {
            cerr << "Empty or null response from server" << endl;
            return false;
        } else {
            cout << "API response success: " << response.dump() << endl;
            return true;
        }
    } catch (const exception& e) {
        cerr << "Error sending event to server: " << e.what() << endl;
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