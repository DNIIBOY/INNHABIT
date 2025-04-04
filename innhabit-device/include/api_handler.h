#ifndef API_HANDLER_H
#define API_HANDLER_H

#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "common.h"

/**
 * @struct ApiEvent
 * @brief Represents an event to be sent to the API.
 */
struct ApiEvent {
    std::string event_type; ///< Type of event (e.g., "entered", "exited").
    std::string timestamp;  ///< Timestamp in ISO 8601 format.
};

/**
 * @class ApiHandler
 * @brief Handles communication with a remote API, including event and image uploads.
 */
class ApiHandler {
public:
    /**
     * @brief Constructor that initializes the API handler.
     * @param config Shared pointer to the configuration object.
     */
    ApiHandler(std::shared_ptr<Configuration> config);

    /**
     * @brief Destructor that ensures proper cleanup.
     */
    ~ApiHandler();

    // Thread management
    /**
     * @brief Starts the API handler processing loop in a separate thread.
     */
    void start();

    /**
     * @brief Signals the handler to stop processing and clean up resources.
     */
    void shutdown();

    /**
     * @brief Waits for the API handler thread to finish execution.
     */
    void join();

    // Event handling
    /**
     * @brief Queues a person-related event for API submission.
     * @param event_type Type of event (e.g., "entered", "exited").
     * @return True if the event was successfully queued, false otherwise.
     */
    bool onPersonEvent(const std::string& event_type);

    /**
     * @brief Sends an image to the API server.
     * @param image The OpenCV image to send.
     * @return True if the image was successfully uploaded, false otherwise.
     */
    bool sendImage(const cv::Mat& image);

    // Utility functions
    /**
     * @brief Saves an API response to a file.
     * @param response The JSON response data.
     * @param filename The name of the file to save the response to.
     */
    void saveResponseToFile(const nlohmann::json& response, const std::string& filename);

private:
    using Json = nlohmann::json;

    // Initialization and processing
    /**
     * @brief Initializes the API handler (e.g., setting up CURL).
     */
    void initialize();

    /**
     * @brief Processes queued events in the event queue.
     */
    void processEvents();

    /**
     * @brief Processes a single API event.
     * @param event The event to process.
     */
    void processSingleEvent(const ApiEvent& event);

    /**
     * @brief Processes any remaining events in the queue before exiting.
     */
    void processRemainingEvents();

    // HTTP request handling
    /**
     * @brief Sends a POST request to the API.
     * @param endpoint The API endpoint to send the request to.
     * @param data The JSON data to send.
     * @return The JSON response from the server.
     */
    nlohmann::json sendPostRequest(const std::string& endpoint, const nlohmann::json& data);

    /**
     * @brief Callback function for handling CURL response data.
     * @param contents Pointer to the received data.
     * @param size Size of one data element.
     * @param nmemb Number of elements.
     * @param data Pointer to the output string to store the response.
     * @return The number of bytes written.
     */
    static size_t writeCallback(void* contents, size_t size, size_t nmemb, std::string* data);

    // Time utilities
    /**
     * @brief Gets the current timestamp in ISO 8601 format.
     * @return A string representing the current timestamp.
     */
    std::string getTimestampISO();

    // Member variables
    std::string api_key_;  ///< API authentication key.
    std::string base_url_; ///< Base URL of the API.
    CURL* curl_;           ///< CURL handle for making HTTP requests.
    bool should_exit_;     ///< Flag to indicate if the handler should stop.

    std::thread thread_;                 ///< Thread for processing API events.
    std::queue<ApiEvent> event_queue_;   ///< Queue for storing API events.
    std::mutex queue_mutex_;             ///< Mutex for thread-safe event queue access.
    std::condition_variable queue_cv_;   ///< Condition variable for event processing synchronization.
};

#endif // API_HANDLER_H
