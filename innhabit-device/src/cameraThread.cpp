#include "cameraThread.h"
#include <iostream>

// Constructor for the FrameCapturer class
FrameCapturer::FrameCapturer(std::queue<cv::Mat>& frameQueue, 
                           std::mutex& frameMutex,
                           std::condition_variable& frameCV,
                           std::atomic<bool>& shouldExit,
                           const int maxQueueSize) 
    : m_frameQueue(frameQueue),
      m_frameMutex(frameMutex),
      m_frameCV(frameCV),
      m_shouldExit(shouldExit),
      m_maxQueueSize(maxQueueSize) {
}

// Start the thread
void FrameCapturer::start(cv::VideoCapture& cap) {
    m_thread = std::thread(&FrameCapturer::captureFrames, this, std::ref(cap));
}

// Wait for the thread to finish
void FrameCapturer::join() {
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

// Capture frames from the camera and add them to the frame queue to be processed by other threads
void FrameCapturer::captureFrames(cv::VideoCapture& cap) {
    while (!m_shouldExit) {
        cv::Mat frame;
        cap.read(frame);
        {
            std::unique_lock<std::mutex> lock(m_frameMutex);
            m_frameCV.wait(lock, [this] { 
                return m_frameQueue.size() < m_maxQueueSize || m_shouldExit; 
            });
            
            if (m_shouldExit) {
                m_frameCV.notify_all();
                break;
            }
            
            if (m_frameQueue.size() >= m_maxQueueSize) {
                continue; // Drop frame
            }
            m_frameQueue.push(frame.clone());
        }
        
        m_frameCV.notify_one();
    }
}