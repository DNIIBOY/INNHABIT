#ifndef RTSP_STREAM_MANAGER_H
#define RTSP_STREAM_MANAGER_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

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

RTSPStreamManager::RTSPStreamManager(std::queue<cv::Mat>& displayQueue, 
                                   std::mutex& displayMutex,
                                   std::condition_variable& displayCV,
                                   std::atomic<bool>& shouldExit) 
    : m_displayQueue(displayQueue),
      m_displayMutex(displayMutex),
      m_displayCV(displayCV),
      m_shouldExit(shouldExit) {
}

void RTSPStreamManager::start() {
    m_thread = std::thread(&RTSPStreamManager::displayFrames, this);
}

void RTSPStreamManager::join() {
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void RTSPStreamManager::displayFrames() {
    auto lastDetectionTime = std::chrono::steady_clock::now();
    const int fpsBufferSize = 16;
    float fpsBuffer[fpsBufferSize] = {0.0};
    int frameCount = 0;
    
    // GStreamer pipeline for RTSP streaming
    const std::string gstreamerPipeline = 
    "appsrc ! videoconvert ! "
    "x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast ! "
    "h264parse ! "
    "rtph264pay config-interval=1 pt=96 ! "
    "udpsink host=127.0.0.1 port=8000";  // Note: Using RTP port 8000
    
    cv::VideoWriter writer;
    bool writerInitialized = false;
    const double outputFps = 30.0;
    cv::Size frameSize;
    
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
                frameSize = frame.size();
            }
        }
        m_displayCV.notify_one();
        
        if (!frame.empty()) {
            // FPS calculation
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
            
            cv::putText(frame, cv::format("FPS: %.2f", avgFps), cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // Initialize writer if not already done
            if (!writerInitialized && !frame.empty()) {
                writer.open(gstreamerPipeline, cv::CAP_GSTREAMER, 0, outputFps, frameSize, true);
                if (!writer.isOpened()) {
                    std::cerr << "Failed to open GStreamer writer. Ensure GStreamer is installed and the RTSP server is running." << std::endl;
                    // You might want to retry after a delay instead of breaking
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    continue;
                }
                std::cout << "GStreamer writer initialized successfully" << std::endl;
                writerInitialized = true;
            }
            
            // Write frame if writer is initialized
            if (writerInitialized) {
                writer.write(frame);
            }
        }
    }
    
    if (writerInitialized) {
        writer.release();
        std::cout << "GStreamer writer released" << std::endl;
    }
}

#endif // RTSP_STREAM_MANAGER_H