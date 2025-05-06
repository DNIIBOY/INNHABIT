#ifndef DETECTION_PROCESSOR_H
#define DETECTION_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include "detector.h"
#include "tracker/people_tracker.h"

/**
 * @class DetectionProcessor
 * @brief Processes frames for object detection and tracking.
 *
 * This class handles the processing of frames, where it performs object detection 
 * using the provided detector and tracks detected people using the provided tracker.
 * The processed frames are then added to a display queue for rendering.
 */
class DetectionProcessor {
public:
    /**
     * @brief Constructs a DetectionProcessor object.
     *
     * @param frameQueue A reference to the queue of frames to be processed.
     * @param frameMutex A reference to the mutex used to protect the frame queue.
     * @param frameCV A reference to the condition variable used to synchronize frame processing.
     * @param displayQueue A reference to the queue of frames to be displayed.
     * @param displayMutex A reference to the mutex used to protect the display queue.
     * @param displayCV A reference to the condition variable used to synchronize frame display.
     * @param shouldExit A reference to the atomic boolean indicating whether to stop processing.
     * @param maxQueueSize The maximum size of the display queue.
     */
    DetectionProcessor(std::queue<cv::Mat>& frameQueue, 
                       std::mutex& frameMutex,
                       std::condition_variable& frameCV,
                       std::queue<cv::Mat>& displayQueue,
                       std::mutex& displayMutex,
                       std::condition_variable& displayCV,
                       std::atomic<bool>& shouldExit,
                       const int maxQueueSize);

    /**
     * @brief Starts the frame processing in a separate thread.
     *
     * This method initializes a thread to begin processing frames using the provided
     * detector and tracker. The thread will process frames as long as `shouldExit` is false.
     *
     * @param detector A pointer to the GenericDetector used for detecting objects in the frames.
     * @param tracker A reference to the PeopleTracker used for tracking detected people.
     */
    void start(GenericDetector* detector, tracker::PeopleTracker& tracker);

    /**
     * @brief Joins the processing thread.
     *
     * This method waits for the processing thread to finish execution.
     */
    void join();

private:
    /**
     * @brief The function that processes frames in the thread.
     *
     * This method processes frames by performing object detection and tracking,
     * then adds the processed frames to the display queue.
     *
     * @param detector A pointer to the GenericDetector used for detecting objects in the frames.
     * @param tracker A reference to the PeopleTracker used for tracking detected people.
     */
    void processFrames(GenericDetector* detector, tracker::PeopleTracker& tracker);

    std::queue<cv::Mat>& m_frameQueue; ///< A reference to the queue containing the frames to be processed.
    std::mutex& m_frameMutex; ///< A reference to the mutex protecting the frame queue.
    std::condition_variable& m_frameCV; ///< A reference to the condition variable for synchronizing frame processing.
    std::queue<cv::Mat>& m_displayQueue; ///< A reference to the queue containing the processed frames to be displayed.
    std::mutex& m_displayMutex; ///< A reference to the mutex protecting the display queue.
    std::condition_variable& m_displayCV; ///< A reference to the condition variable for synchronizing frame display.
    std::atomic<bool>& m_shouldExit; ///< A reference to the atomic boolean indicating whether processing should stop.
    const int m_maxQueueSize; ///< The maximum size of the display queue.
    std::thread m_thread; ///< The thread used to process frames.
};

#endif // DETECTION_PROCESSOR_H
