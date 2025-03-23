#include "api_handler.h"
#include "detector.h"
#include "tracker.h"
#include "common.h"
#include "cameraThread.h"
#include "detectionThread.h"
#include "displayThread.h"
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
const int MAX_QUEUE_SIZE = 10;

// Shared resources for threads
std::queue<Mat> frameQueue; // Queue for frames captured from camera
std::queue<Mat> displayQueue;
std::mutex frameMutex; // Mutex for frame queue so threads can lock it
std::mutex displayMutex;
std::condition_variable frameCV; // Condition variable to notify threads of new frames
std::condition_variable displayCV;
std::atomic<bool> shouldExit(false); // Flag to notify threads to exit if something goes wrong
std::unique_ptr<ApiHandler> apiHandler;

// Movement event callback function for tracker when person event happens
void movementEventCallback(const TrackedPerson& person, const std::string& eventType) {
    if (apiHandler) {
        apiHandler->onPersonEvent(eventType);
    } else {
        ERROR("API handler not initialized");
    }
}

void printUsage(const char* progName) {
    cout << "Usage: " << progName << " [--video <video_file>] [--image <image_file>]" << endl;
    cout << "  --video <file> : Process a video file" << endl;
    cout << "  (No arguments defaults to webcam)" << endl;
}

// Load settings from a JSON file
Config loadSettings(const string& settingsFile) {
    try {
        config = loadConfig(settingsFile);
        LOG("Loaded config for device: " << config.deviceName);
        return config;
    } catch (const std::exception& e) {
        ERROR("Error loading config: " << e.what());
        return Config();
    }
}

int main(int argc, char** argv) {
    config = loadSettings("../settings.json");
    PeopleTracker tracker;

    // Set up tracker with entrance zones and movement callback
    tracker.setMovementCallback(movementEventCallback);
    tracker.setEntranceZones(config.entranceZones);

    // Create API handler and start its thread
    apiHandler.reset(new ApiHandler(config.serverEventAPI));
    apiHandler->start();
    
    // Create detector to detect people in frames
    GenericDetector* detector = createDetector(config.modelEnginePath, {"person"});
    if (!detector) {
        cerr << "Error: Failed to initialize detector." << endl;
        return -1;
    }

    // intialize the camera
    VideoCapture cap;
    cap.open(0, CAP_V4L2);
    cap.set(CAP_PROP_AUTO_EXPOSURE, 3);
    //cap.set(CAP_PROP_FRAME_WIDTH, 640);
    //cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    //cap.set(CAP_PROP_FPS, 30);
    
    LOG("Camera opened successfully. Resolution: " 
        << cap.get(CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CAP_PROP_FRAME_HEIGHT) 
        << " framerate: " << cap.get(CAP_PROP_FPS));
    
    LOG("Press 'q' to quit.");

    // Create and start threads
    FrameCapturer capturer(frameQueue, frameMutex, frameCV, shouldExit, MAX_QUEUE_SIZE);
    DetectionProcessor processor(frameQueue, frameMutex, frameCV, 
                              displayQueue, displayMutex, displayCV, 
                              shouldExit, MAX_QUEUE_SIZE);
    DisplayManager display(displayQueue, displayMutex, displayCV, shouldExit);
    
    capturer.start(cap);
    processor.start(detector, tracker);
    display.start();
    
    // Wait for threads to complete
    capturer.join();
    processor.join();
    display.join();
    
    // Properly shutdown the API handler
    apiHandler->shutdown();
    apiHandler->join();
    apiHandler.reset();

    // Cleanup
    cap.release();
    destroyAllWindows();
    delete detector;
    
    return 0;
}