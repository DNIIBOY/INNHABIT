#ifndef DETECTION_PROCESSOR_H
#define DETECTION_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include "detector.h"
#include "tracker.h"

class DetectionProcessor {
public:
    DetectionProcessor(std::queue<cv::Mat>& frameQueue, 
                     std::mutex& frameMutex,
                     std::condition_variable& frameCV,
                     std::queue<cv::Mat>& displayQueue,
                     std::mutex& displayMutex,
                     std::condition_variable& displayCV,
                     std::atomic<bool>& shouldExit,
                     const int maxQueueSize);
    
    void start(GenericDetector* detector, PeopleTracker& tracker);
    void join();

private:
    void processFrames(GenericDetector* detector, PeopleTracker& tracker);
    
    std::queue<cv::Mat>& m_frameQueue;
    std::mutex& m_frameMutex;
    std::condition_variable& m_frameCV;
    std::queue<cv::Mat>& m_displayQueue;
    std::mutex& m_displayMutex;
    std::condition_variable& m_displayCV;
    std::atomic<bool>& m_shouldExit;
    const int m_maxQueueSize;
    std::thread m_thread;
};

#endif // DETECTION_PROCESSOR_H