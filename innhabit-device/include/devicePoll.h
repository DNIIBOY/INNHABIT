#ifndef DEVICE_POLLER_H
#define DEVICE_POLLER_H

#include <string>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <opencv2/opencv.hpp>
#include <curl/curl.h>
#include "api_handler.h"
#include <nlohmann/json.hpp>
#include "common.h" // Include for logging macros

using json = nlohmann::json;

/**
 * @class DevicePoller
 * @brief A class that polls a server for device actions and handles responses.
 * 
 * This class manages periodic polling of a server endpoint, processes responses,
 * and handles actions such as capturing and sending images. It uses CURL for HTTP
 * requests and maintains thread-safe frame queue operations.
 */
class DevicePoller {
public:
    /**
     * @brief Constructor for DevicePoller
     * 
     * @param apiUrl Base URL of the API server
     * @param apiKey Authentication key for API requests
     * @param shouldExit Atomic flag to signal thread termination
     * @param frameQueue Queue for storing captured frames
     * @param frameMutex Mutex for thread-safe frame queue access
     * @param frameCV Condition variable for frame queue synchronization
     * @param apiHandler Pointer to API handler instance
     * @param pollIntervalSeconds Interval between polling requests in seconds
     * @throws std::runtime_error if CURL initialization fails
     */
    DevicePoller(const std::string& apiUrl, 
                 const std::string& apiKey,
                 std::atomic<bool>& shouldExit,
                 std::queue<cv::Mat>& frameQueue,
                 std::mutex& frameMutex,
                 std::condition_variable& frameCV,
                 ApiHandler* apiHandler,
                 int pollIntervalSeconds);

    /**
     * @brief Destructor for DevicePoller
     * 
     * Cleans up CURL resources and ensures proper shutdown.
     */
    ~DevicePoller();

    /**
     * @brief Starts the polling thread
     * 
     * Launches a separate thread that continuously polls the server.
     */
    void start();

    /**
     * @brief Joins the polling thread
     * 
     * Waits for the polling thread to complete execution.
     */
    void join();

private:
    std::string m_apiUrl;                    ///< Base URL for API requests
    std::string m_apiKey;                    ///< API authentication key
    std::atomic<bool>& m_shouldExit;         ///< Reference to exit flag
    std::queue<cv::Mat>& m_frameQueue;       ///< Reference to frame queue
    std::mutex& m_frameMutex;               ///< Reference to frame queue mutex
    std::condition_variable& m_frameCV;     ///< Reference to frame condition variable
    ApiHandler* m_apiHandler;               ///< Pointer to API handler
    int m_pollIntervalSeconds;              ///< Polling interval in seconds
    CURL* m_curl;                           ///< CURL handle for HTTP requests
    std::thread m_thread;                   ///< Polling thread

    /**
     * @brief Main polling loop
     * 
     * Continuously polls the server and processes responses until termination is requested.
     */
    void pollServer();

    /**
     * @brief Captures an image from the frame queue and sends it
     * 
     * @return true if image was successfully captured and sent, false otherwise
     */
    bool captureAndSendImage();

    /**
     * @brief Sends a GET request to the specified endpoint
     * 
     * @param endpoint API endpoint to query
     * @return json JSON response from the server
     */
    json sendGetRequest(const std::string& endpoint);

    /**
     * @brief CURL write callback function
     * 
     * @param contents Data received from CURL
     * @param size Size of each data element
     * @param nmemb Number of data elements
     * @param data Pointer to string to store response
     * @return size_t Number of bytes processed
     */
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data);
};

#endif // DEVICE_POLLER_H