#include "rtspStreamManager.h"
#include <iostream>

RTSPStreamManager::RTSPStreamManager(std::queue<cv::Mat>& displayQueue, 
                                     std::mutex& displayMutex,
                                     std::condition_variable& displayCV,
                                     std::atomic<bool>& shouldExit,
                                     const std::string& rtspPath,
                                     const std::string& host,
                                     int port) 
    : m_displayQueue(displayQueue),
      m_displayMutex(displayMutex),
      m_displayCV(displayCV),
      m_shouldExit(shouldExit),
      m_rtspPath(rtspPath),
      m_host(host),
      m_port(port) {
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
    
    // Updated pipeline using rtspclientsink
    std::string gstreamerPipeline = 
        "appsrc ! videoconvert ! "
        "x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast ! "
        "h264parse ! "
        "rtspclientsink location=rtsp://" + m_host + ":" + std::to_string(m_port) + m_rtspPath;
    
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
                std::cout << "Popped frame: " << frame.cols << "x" << frame.rows << std::endl;
            } else {
                std::cout << "Display queue empty" << std::endl;
                continue;
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
            
            if (!writerInitialized && !frame.empty()) {
                writer.open(gstreamerPipeline, cv::CAP_GSTREAMER, 0, outputFps, frameSize, true);
                if (!writer.isOpened()) {
                    std::cerr << "Failed to open GStreamer writer. Ensure MediaMTX is running on " 
                              << m_host << ":" << m_port << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    continue;
                }
                std::cout << "GStreamer writer initialized successfully for rtsp://" 
                          << m_host << ":" << m_port << m_rtspPath << std::endl;
                writerInitialized = true;
            }
            
            if (writerInitialized) {
                writer.write(frame); // send frame
            } 
        }
    }
    
    if (writerInitialized) {
        writer.release();
        std::cout << "GStreamer writer released" << std::endl;
    }
}