#include "detectionThread.h"

DetectionProcessor::DetectionProcessor(std::queue<cv::Mat>& frameQueue, 
                                     std::mutex& frameMutex,
                                     std::condition_variable& frameCV,
                                     std::queue<cv::Mat>& displayQueue,
                                     std::mutex& displayMutex,
                                     std::condition_variable& displayCV,
                                     std::atomic<bool>& shouldExit,
                                     const int maxQueueSize) 
    : m_frameQueue(frameQueue),
      m_frameMutex(frameMutex),
      m_frameCV(frameCV),
      m_displayQueue(displayQueue),
      m_displayMutex(displayMutex),
      m_displayCV(displayCV),
      m_shouldExit(shouldExit),
      m_maxQueueSize(maxQueueSize) {
}

void DetectionProcessor::start(GenericDetector* detector, tracker::PeopleTracker& tracker) {
    m_thread = std::thread(&DetectionProcessor::processFrames, this, detector, std::ref(tracker));
}

void DetectionProcessor::join() {
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void DetectionProcessor::processFrames(GenericDetector* detector, tracker::PeopleTracker& tracker) {
    while (!m_shouldExit) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(m_frameMutex);
            m_frameCV.wait(lock, [this] { return !m_frameQueue.empty() || m_shouldExit; });
            if (m_shouldExit) break;
            if (!m_frameQueue.empty()) {
                frame = std::move(m_frameQueue.front());
                m_frameQueue.pop();
            }
        }
        m_frameCV.notify_one();
        
        if (!frame.empty()) {
            detector->detect(frame);
            tracker.update(detector->getDetections(), frame.rows);
            tracker.draw(frame);
            {
                std::unique_lock<std::mutex> lock(m_displayMutex);
                m_displayCV.wait(lock, [this] { 
                    return m_displayQueue.size() < m_maxQueueSize || m_shouldExit; 
                });
                
                if (m_shouldExit) break;
                
                m_displayQueue.push(frame.clone());
            }
            
            m_displayCV.notify_one();
        }
    }
}