#include "displayThread.h"
#include <opencv2/highgui/highgui.hpp>
#include "api_handler.h" // Include to access apiHandler
#include "common.h" // Include for logging macros

// External declarations for global variables from main.cpp
extern std::queue<cv::Mat> frameQueue;
extern std::mutex frameMutex;
extern std::condition_variable frameCV;
extern std::unique_ptr<ApiHandler> apiHandler;
extern std::atomic<bool> shouldExit;

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
            // Calculate FPS
            auto currentTime = std::chrono::steady_clock::now();
            float frameTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastDetectionTime).count();
            lastDetectionTime = currentTime;
            
            float fps = (frameTimeMs > 0) ? std::min(1000.0f / frameTimeMs, 30.0f) : 0.0f;
            fpsBuffer[frameCount % fpsBufferSize] = fps;
            frameCount++;
            
            float avgFps = 0.0;
            for (int i = 0; i < std::min(frameCount, fpsBufferSize); i++) {
                avgFps += fpsBuffer[i];
            }
            avgFps /= std::min(frameCount, fpsBufferSize);
            
            std::string fpsText = cv::format("FPS: %.2f", avgFps);
            cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            cv::imshow("People Detection and Tracking", frame);
            
            int key = cv::waitKey(1);
            if (key == 'q') { // Quit on 'q'
                m_shouldExit = true;
                break;
            } else if (key == 32) { // Spacebar pressed (ASCII 32)
                if (apiHandler) {
                    LOG("Spacebar pressed, fetching latest frame from camera thread...");

                    cv::Mat newFrame;
                    {
                        std::unique_lock<std::mutex> lock(frameMutex);
                        // Wait briefly for a frame if the queue is empty
                        frameCV.wait_for(lock, std::chrono::milliseconds(100), [] { 
                            return !frameQueue.empty(); 
                        });

                        if (!frameQueue.empty()) {
                            newFrame = frameQueue.front(); // Get the latest frame
                            // Note: We don't pop it here to avoid interfering with DetectionProcessor
                        } else {
                            ERROR("No frame available in frameQueue");
                            continue;
                        }
                    }

                    if (!newFrame.empty()) {
                        apiHandler->sendImage(newFrame);
                    } else {
                        ERROR("Captured frame is empty");
                    }
                } else {
                    ERROR("API handler not initialized, cannot send image");
                }
            }
        }
    }
    
    cv::destroyAllWindows();
}