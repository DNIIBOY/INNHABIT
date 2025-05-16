#include "cameraThread.h"
#include <iostream>
#include <chrono>

// Constructor for the FrameCapturer class
FrameCapturer::FrameCapturer(std::queue<cv::Mat>& frameQueue, 
                           std::mutex& frameMutex, std::queue<cv::Mat>& displayQueue, std::mutex& displayMutex,
                           std::condition_variable& frameCV,
                           std::atomic<bool>& shouldExit,
                           const int maxQueueSize) 
    : m_frameQueue(frameQueue),
      m_frameMutex(frameMutex),
      m_displayQueue(displayQueue),
      m_displayMutex(displayMutex),
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
	int brightness_index = 0;
	double brightness = 200;
	cv::Mat gray_frame;
    while (!m_shouldExit) {
	
        cv::Mat frame;
        cap.read(frame);
	brightness_index += 1;
	if (brightness_index >= 30)
	{
		cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
		brightness = cv::mean(gray_frame)[0];
		std::cout << "brightness" << brightness << "\n";
		brightness_index = 0;
	}
		
		
		
	
        {	
            std::unique_lock<std::mutex> lock(m_frameMutex);
            m_frameCV.wait(lock, [this] { 
                return m_frameQueue.size() < m_maxQueueSize || m_shouldExit; 
            });
            
            if (m_shouldExit) {
                m_frameCV.notify_all();
                break;
            }
            
            if (m_frameQueue.size() >= m_maxQueueSize || brightness < 10.0) {
                continue; // Drop frame
            }
            if ( brightness < 10.0){
                std::unique_lock<std::mutex> lock(m_displayMutex);
                m_displayQueue.push(frame.clone());
            }
            m_frameQueue.push(frame.clone());
        }
        
        m_frameCV.notify_one();
    }
}
