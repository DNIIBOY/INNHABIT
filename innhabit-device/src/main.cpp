// local libs
#include "api_handler.h"
#include "detector.h"
#include "tracker.h"
#include "common.h"
#include "cameraThread.h"
#include "detectionThread.h"
#include "rtspStreamManager.h"  // Updated include
#include "displayThread.h"
#include "devicePoll.h"
// libs
#include <iostream>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

// Global variables
Config config;
/**
 * Defines the max size of frameQueue and displayQueue
*/
const int MAX_QUEUE_SIZE = 5;

/**
    * Stores frames captured by FrameCapturer
*/
std::queue<Mat> frameQueue;
/**
    * Stores processed frames after inference and tracking
*/
std::queue<Mat> displayQueue;
/**
    * Mutex for threads to lock frameQueue
*/
std::mutex frameMutex;
/**
    * Mutex for threads to lock displayQueue
*/
std::mutex displayMutex;
/**
    * Condition variable to notify threads of new frames ready for processing
*/
std::condition_variable frameCV; // Condition variable to notify threads of new frames
/**
    * Condition variable to notify threads of new frames ready to be displayThread
*/
std::condition_variable displayCV;
/**
    * Atmoix variable Flag used to notify threads to exit if something goes wrong
*/
std::atomic<bool> shouldExit(false); // Flag to notify threads to exit if something goes wrong
std::unique_ptr<ApiHandler> apiHandler;

/**
* Movement event callback function for tracker when person event happens
* ex. person exiting or entering exit zones
* Calls apihandler to send event to server
*/
void movementEventCallback(const TrackedPerson& person, const std::string& eventType) {
    if (apiHandler) {
        apiHandler->onPersonEvent(eventType);
    } else {
        ERROR("API handler not initialized");
    }
}

/**
* Prints the usage guides for the program.
*/
void printUsage(const char* progName) {
    cout << "Usage: " << progName << " [--video <video_file>] [--image <image_file>]" << endl;
    cout << "  --video <file> : Process a video file" << endl;
    cout << "  (No arguments defaults to webcam)" << endl;
}

int main(int argc, char** argv) {
    config = loadConfig("../settings.json");
    PeopleTracker tracker;

    // Set up tracker with entrance zones and movement callback
    tracker.setMovementCallback(movementEventCallback);
    tracker.setEntranceZones(config.entranceZones);

    // Create API handler and start its thread
    apiHandler.reset(new ApiHandler(config.serverAPI, config.serverApikey));
    apiHandler->start();
    
    // Create detector to detect people in frames
    GenericDetector* detector = createDetector(config.modelPath, {"person"});
    if (!detector) {
        cerr << "Error: Failed to initialize detector." << endl;
        return -1;
    }

    std::string pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! nvvidconv flip-method=2 ! videoconvert ! video/x-raw, format=BGR ! appsink";
    LOG("Attempting to open pipeline: " << pipeline);
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }
    
    LOG("Camera opened successfully. Resolution: " 
        << cap.get(CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CAP_PROP_FRAME_HEIGHT) 
        << " framerate: " << cap.get(CAP_PROP_FPS));

    // Create and start threads
    FrameCapturer capturer(frameQueue, frameMutex, frameCV, shouldExit, MAX_QUEUE_SIZE);
    DetectionProcessor processor(frameQueue, frameMutex, frameCV, 
                              displayQueue, displayMutex, displayCV, 
                              shouldExit, MAX_QUEUE_SIZE);
    DevicePoller poller(config.serverAPI, config.serverApikey, shouldExit, 
                        frameQueue, frameMutex, frameCV, apiHandler.get(), 60); // Poll every 5 seconds
    //RTSPStreamManager streamManager(displayQueue, displayMutex, displayCV, shouldExit, "/stream", "127.0.0.1");  // Updated class name
    DisplayManager display(displayQueue, displayMutex, displayCV, shouldExit);

    capturer.start(cap);
    processor.start(detector, tracker);
    //streamManager.start();  // Updated variable name
    display.start();
    poller.start();
    // Wait for threads to complete
    capturer.join();
    processor.join();
    //streamManager.join();  // Updated variable name
    display.join();
    poller.join();
    // Properly shutdown the API handler
    apiHandler->shutdown();
    apiHandler->join();
    apiHandler.reset();

    // Cleanup
    cap.release();
    delete detector;
    
    return 0;
}
