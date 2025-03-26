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
    int frameCount = 0;
    
    std::string gstreamerPipeline = 
        "appsrc ! videoconvert ! "
        "nvvidconv ! "
        "nvv4l2h264enc bitrate=300000 preset-level=1 iframeinterval=15 maxperf-enable=true ! "
        "h264parse ! "
        "rtspclientsink location=rtsp://" + m_host + ":" + std::to_string(m_port) + m_rtspPath;
    
    cv::VideoWriter writer;
    bool writerInitialized = false;
    const double outputFps = 15.0;
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
                frame = std::move(m_displayQueue.front());
                m_displayQueue.pop();
                frameSize = frame.size();
            } else {
                continue;
            }
        }
        m_displayCV.notify_one();
        
        if (!frame.empty()) {
            cv::Mat resizedFrame;
            cv::resize(frame, resizedFrame, cv::Size(640, 360), 0, 0, cv::INTER_LINEAR);
            frame = resizedFrame;

            if (frameCount % 10 == 0) {
                auto currentTime = std::chrono::steady_clock::now();
                float frameTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastDetectionTime).count();
                lastDetectionTime = currentTime;
                float fps = (frameTimeMs > 0) ? 1000.0f / frameTimeMs : 0.0f;
                cv::putText(frame, cv::format("FPS: %.2f", fps), cv::Point(10, 30), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }
            frameCount++;

            if (!writerInitialized) {
                writer.open(gstreamerPipeline, cv::CAP_GSTREAMER, 0, outputFps, frame.size(), true);
                if (!writer.isOpened()) {
                    std::cerr << "Failed to open GStreamer writer." << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    continue;
                }
                std::cout << "GStreamer writer initialized successfully." << std::endl;
                writerInitialized = true;
            }
            
            if (writerInitialized && frameCount % 2 == 0) {
                writer.write(frame);
            }
        }
    }
    
    if (writerInitialized) {
        writer.release();
        std::cout << "GStreamer writer released." << std::endl;
    }
}