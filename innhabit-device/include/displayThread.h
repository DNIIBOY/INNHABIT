#ifndef DISPLAY_MANAGER_H
#define DISPLAY_MANAGER_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <thread>

/**
 * @class DisplayManager
 * @brief Manages the display of labeled and tracked images.
 */
class DisplayManager {
public:
    /**
     * @brief Constructor for DisplayManager.
     * @param displayQueue Reference to the queue holding frames to be displayed.
     * @param displayMutex Mutex to synchronize access to the queue.
     * @param displayCV Condition variable to signal frame availability.
     * @param shouldExit Atomic flag to indicate when to stop processing.
     */
    DisplayManager(std::queue<cv::Mat>& displayQueue, 
                  std::mutex& displayMutex,
                  std::condition_variable& displayCV,
                  std::atomic<bool>& shouldExit);
    
    /**
     * @brief Starts a thread displaying labeled images from the queue.
     */
    void start();
    
    /**
     * @brief Joins the DisplayManager thread for safe termination.
     */
    void join();

private:
    /**
     * @brief Function that runs in a separate thread to display frames.
     */
    void displayFrames();
    
    std::queue<cv::Mat>& m_displayQueue;           ///< Queue of images to display
    std::mutex& m_displayMutex;                    ///< Mutex for synchronizing access to the queue
    std::condition_variable& m_displayCV;          ///< Condition variable for signaling frame availability
    std::atomic<bool>& m_shouldExit;               ///< Atomic flag to signal termination
    std::thread m_thread;                          ///< Thread for managing the display process
};

#endif // DISPLAY_MANAGER_H
