#ifndef CAMERA_THREAD_H
#define CAMERA_THREAD_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <thread>

class FrameCapturer {
public:
    FrameCapturer(std::queue<cv::Mat>& frameQueue, 
                 std::mutex& frameMutex,
                 std::condition_variable& frameCV,
                 std::atomic<bool>& shouldExit,
                 const int maxQueueSize);
    
    void start(cv::VideoCapture& cap);
    void join();

private:
    void captureFrames(cv::VideoCapture& cap);
    
    std::queue<cv::Mat>& m_frameQueue;
    std::mutex& m_frameMutex;
    std::condition_variable& m_frameCV;
    std::atomic<bool>& m_shouldExit;
    const int m_maxQueueSize;
    std::thread m_thread;
};

#endif // CAMERA_THREAD_H