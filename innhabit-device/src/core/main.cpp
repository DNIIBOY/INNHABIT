#include "api_handler.h"
#include "detector.h"
#include <tracker/people_tracker.h>
#include "common.h"
#include "cameraThread.h"
#include "detectionThread.h"
#include "rtspStreamManager.h"
#include "displayThread.h"
#include "devicePoll.h"
#include <iostream>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

const int MAX_QUEUE_SIZE = 5;

std::queue<cv::Mat> frameQueue;
std::queue<cv::Mat> displayQueue;
std::mutex frameMutex;
std::mutex displayMutex;
std::condition_variable frameCV;
std::condition_variable displayCV;
std::atomic<bool> shouldExit(false);
std::unique_ptr<ApiHandler> apiHandler;

void movementEventCallback(const tracker::TrackedPerson& person, const std::string& eventType) {
    if (apiHandler) {
        apiHandler->onPersonEvent(eventType);
    } else {
        std::cerr << "[ERROR] API handler not initialized" << std::endl;
    }
}

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [--video <video_file>] [--image <image_file>]" << std::endl;
    std::cout << "  --video <file> : Process a video file" << std::endl;
    std::cout << "  (No arguments defaults to webcam)" << std::endl;
}

int main(int argc, char** argv) {
    auto config = Configuration::LoadFromFile("../settings.json");

    tracker::PeopleTracker tracker(config);
    tracker.set_movement_callback(movementEventCallback);
    
    apiHandler.reset(new ApiHandler(config));
    apiHandler->start();
    
    GenericDetector* detector = createDetector(config->GetModelPath(), {"person"});
    if (!detector) {
        std::cerr << "Error: Failed to initialize detector." << std::endl;
        return -1;
    }
    
    std::string pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! nvvidconv flip-method=2 ! videoconvert ! video/x-raw, format=BGR ! appsink";
    LOG("Attempting to open pipeline: " << pipeline);
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    cap.set(cv::CAP_PROP_FPS, 30);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }
    std::cout << "[INFO] Camera opened successfully. Resolution: "
              << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x" << cap.get(cv::CAP_PROP_FRAME_HEIGHT)
              << " framerate: " << cap.get(cv::CAP_PROP_FPS) << std::endl;
    
    FrameCapturer capturer(frameQueue, frameMutex, frameCV, shouldExit, MAX_QUEUE_SIZE);
    DetectionProcessor processor(frameQueue, frameMutex, frameCV, 
                                displayQueue, displayMutex, displayCV, 
                                shouldExit, MAX_QUEUE_SIZE);
    DevicePoller poller(config->GetServerApi(), config->GetServerApiKey(), shouldExit, 
                        frameQueue, frameMutex, frameCV, apiHandler.get(), config, 10);
    RTSPStreamManager streamManager(displayQueue, displayMutex, displayCV, shouldExit, "/stream", "127.0.0.1");
    capturer.start(cap);
    processor.start(detector, tracker);
    poller.start();
    streamManager.start();
    capturer.join();
    processor.join();
    poller.join();
    streamManager.join();
    apiHandler->shutdown();
    apiHandler->join();
    apiHandler.reset();
    cap.release();
    delete detector;

    return 0;
}