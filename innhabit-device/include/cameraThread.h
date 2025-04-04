#ifndef CAMERA_THREAD_H
#define CAMERA_THREAD_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <thread>

/**
 * @class FrameCapturer
 * @brief Captures frames from a video stream using OpenCV's VideoCapture.
 *
 * This class is responsible for capturing frames from a video source, such as a camera, 
 * and placing them into a queue for further processing. The capturing runs in a separate thread.
 */
class FrameCapturer {
public:
    /**
     * @brief Constructs a FrameCapturer object.
     *
     * Initializes the FrameCapturer object with references to the frame queue, mutex, condition variable,
     * and other necessary parameters. The frame queue is used to store captured frames for further processing.
     * The maximum size of the queue is also set here.
     *
     * @param frameQueue A reference to the queue where captured frames will be stored.
     * @param frameMutex A reference to the mutex used for thread-safe access to the frame queue.
     * @param frameCV A reference to the condition variable used for synchronizing frame capturing.
     * @param shouldExit A reference to an atomic boolean that signals whether the capturing should stop.
     * @param maxQueueSize The maximum size of the frame queue before dropping frames.
     */
    FrameCapturer(std::queue<cv::Mat>& frameQueue, 
                  std::mutex& frameMutex,
                  std::condition_variable& frameCV,
                  std::atomic<bool>& shouldExit,
                  const int maxQueueSize);

    /**
     * @brief Starts capturing frames in a separate thread.
     *
     * This method initializes a separate thread that continuously captures frames from the 
     * provided `cv::VideoCapture` object and adds them to the `frameQueue`.
     *
     * @param cap A reference to the OpenCV VideoCapture object used for capturing frames.
     */
    void start(cv::VideoCapture& cap);

    /**
     * @brief Waits for the capturing thread to finish.
     *
     * This method ensures that the capturing thread finishes execution safely by joining it.
     */
    void join();

private:
    /**
     * @brief Captures frames from the camera and stores them in the frame queue.
     *
     * This method continuously captures frames from the provided `cv::VideoCapture` object and places
     * them in the `frameQueue`. The method also ensures thread-safe operations on the queue and handles 
     * synchronization using the mutex and condition variable.
     *
     * @param cap A reference to the OpenCV VideoCapture object used to capture frames.
     */
    void captureFrames(cv::VideoCapture& cap);
    
    std::queue<cv::Mat>& m_frameQueue; ///< A reference to the queue storing captured frames.
    std::mutex& m_frameMutex; ///< A reference to the mutex for synchronizing access to the frame queue.
    std::condition_variable& m_frameCV; ///< A reference to the condition variable for synchronizing frame capture.
    std::atomic<bool>& m_shouldExit; ///< A reference to the atomic boolean indicating whether to stop capturing frames.
    const int m_maxQueueSize; ///< The maximum size of the frame queue before dropping frames.
    std::thread m_thread; ///< The thread responsible for capturing frames.
};

#endif // CAMERA_THREAD_H
