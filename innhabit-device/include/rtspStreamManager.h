#ifndef RTSP_STREAM_MANAGER_H
#define RTSP_STREAM_MANAGER_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <string>

/**
 * @class RTSPStreamManager
 * @brief Manages the streaming of video frames to an RTSP server using GStreamer.
 *
 * This class takes frames from a shared queue and streams them to an RTSP server using 
 * GStreamer. It handles synchronization with condition variables and ensures proper 
 * thread management for efficient streaming.
 */
class RTSPStreamManager {
public:
    /**
     * @brief Constructs an RTSPStreamManager object.
     * 
     * @param displayQueue Reference to a queue holding frames to be displayed.
     * @param displayMutex Reference to a mutex for synchronizing access to the frame queue.
     * @param displayCV Reference to a condition variable for notifying when new frames are available.
     * @param shouldExit Reference to an atomic flag to signal when the streaming should stop.
     * @param rtspPath Path to the RTSP stream (default: "/stream").
     * @param host Host IP address of the RTSP server (default: "127.0.0.1").
     * @param port Port number of the RTSP server (default: 8554).
     */
    RTSPStreamManager(std::queue<cv::Mat>& displayQueue, 
                      std::mutex& displayMutex,
                      std::condition_variable& displayCV,
                      std::atomic<bool>& shouldExit,
                      const std::string& rtspPath = "/stream",
                      const std::string& host = "127.0.0.1",
                      int port = 8554);

    /**
     * @brief Starts the RTSP stream in a separate thread.
     *
     * This function spawns a new thread that continuously processes and streams video frames.
     */
    void start();

    /**
     * @brief Joins the RTSP stream thread.
     *
     * This function ensures that the streaming thread is properly joined before the object is destroyed.
     */
    void join();
    
private:
    /**
     * @brief Captures and streams frames from the queue to the RTSP server.
     *
     * This function continuously retrieves frames from the queue, resizes them, overlays FPS data, 
     * and streams them using GStreamer.
     */
    void displayFrames();
    
    std::queue<cv::Mat>& m_displayQueue;  ///< Reference to the shared queue holding frames for streaming.
    std::mutex& m_displayMutex;           ///< Reference to the mutex for synchronizing queue access.
    std::condition_variable& m_displayCV; ///< Reference to the condition variable for queue notifications.
    std::atomic<bool>& m_shouldExit;      ///< Atomic flag indicating whether the streaming should stop.
    std::thread m_thread;                 ///< Thread responsible for streaming frames.

    std::string m_rtspPath;  ///< RTSP stream path (e.g., "/stream").
    std::string m_host;      ///< RTSP server host address.
    int m_port;              ///< RTSP server port number.
};

#endif // RTSP_STREAM_MANAGER_H
