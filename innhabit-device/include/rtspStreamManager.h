#ifndef RTSP_STREAM_MANAGER_H
#define RTSP_STREAM_MANAGER_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <opencv2/opencv.hpp>

class RTSPStreamManager {
public:
    RTSPStreamManager(std::queue<cv::Mat>& displayQueue,
                     std::mutex& displayMutex,
                     std::condition_variable& displayCV,
                     std::atomic<bool>& shouldExit);
    
    void start();
    void join();

private:
    void displayFrames();
    
    std::queue<cv::Mat>& m_displayQueue;
    std::mutex& m_displayMutex;
    std::condition_variable& m_displayCV;
    std::atomic<bool>& m_shouldExit;
    std::thread m_thread;
};

#endif