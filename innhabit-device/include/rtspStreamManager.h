#ifndef RTSP_STREAM_MANAGER_H
#define RTSP_STREAM_MANAGER_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <string>  // Added for std::string

class RTSPStreamManager {
public:
    RTSPStreamManager(std::queue<cv::Mat>& displayQueue, 
                      std::mutex& displayMutex,
                      std::condition_variable& displayCV,
                      std::atomic<bool>& shouldExit,
                      const std::string& rtspPath = "/stream",
                      const std::string& host = "127.0.0.1",
                      int port = 8554);
    
    void start();
    void join();
    
private:
    void displayFrames();
    
    std::queue<cv::Mat>& m_displayQueue;
    std::mutex& m_displayMutex;
    std::condition_variable& m_displayCV;
    std::atomic<bool>& m_shouldExit;
    std::thread m_thread;
    std::string m_rtspPath;  // e.g., "/stream"
    std::string m_host;      // e.g., "127.0.0.1"
    int m_port;              // e.g., 8554
};

#endif // RTSP_STREAM_MANAGER_H
