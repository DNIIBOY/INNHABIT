#ifndef DISPLAY_MANAGER_H
#define DISPLAY_MANAGER_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <thread>

class DisplayManager {
public:
    DisplayManager(std::queue<cv::Mat>& displayQueue, 
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

#endif // DISPLAY_MANAGER_H