#include "displayThread.h"
#include <opencv2/highgui/highgui.hpp>

DisplayManager::DisplayManager(std::queue<cv::Mat>& displayQueue, 
                             std::mutex& displayMutex,
                             std::condition_variable& displayCV,
                             std::atomic<bool>& shouldExit) 
    : m_displayQueue(displayQueue),
      m_displayMutex(displayMutex),
      m_displayCV(displayCV),
      m_shouldExit(shouldExit) {
}

void DisplayManager::start() {
    m_thread = std::thread(&DisplayManager::displayFrames, this);
}

void DisplayManager::join() {
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void DisplayManager::displayFrames() {
    auto lastDetectionTime = std::chrono::steady_clock::now();
    const int fpsBufferSize = 16;
    float fpsBuffer[fpsBufferSize] = {0.0};
    int frameCount = 0;
    
    while (!m_shouldExit) {
        cv::Mat frame;
        
        {
            std::unique_lock<std::mutex> lock(m_displayMutex);
            m_displayCV.wait(lock, [this] { 
                return !m_displayQueue.empty() || m_shouldExit; 
            });
            
            if (m_shouldExit) break;
            
            if (!m_displayQueue.empty()) {
                frame = m_displayQueue.front();
                m_displayQueue.pop();
            }
        }
        
        m_displayCV.notify_one();
        
        if (!frame.empty()) {
            // Calculate FPS based on time since last frame was processed by detection
            auto currentTime = std::chrono::steady_clock::now();
            float frameTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastDetectionTime).count();
            lastDetectionTime = currentTime; // Update to current frame's dequeue time
            
            // Calculate FPS, cap at realistic max (e.g., camera's 30 FPS)
            float fps = (frameTimeMs > 0) ? std::min(1000.0f / frameTimeMs, 30.0f) : 0.0f;
            fpsBuffer[frameCount % fpsBufferSize] = fps;
            frameCount++;
            
            // Compute average FPS
            float avgFps = 0.0;
            for (int i = 0; i < std::min(frameCount, fpsBufferSize); i++) {
                avgFps += fpsBuffer[i];
            }
            avgFps /= std::min(frameCount, fpsBufferSize);
            
            std::string fpsText = cv::format("FPS: %.2f", avgFps);
            cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            cv::imshow("People Detection and Tracking", frame);
            
            if (cv::waitKey(1) == 'q') {
                m_shouldExit = true;
                break;
            }
        }
    }
    
    cv::destroyAllWindows();
}