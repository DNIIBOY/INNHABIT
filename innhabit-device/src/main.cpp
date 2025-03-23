#include "api_handler.h"
#include "detector.h"
#include "tracker.h"
#include "common.h"
#include <iostream>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

Config config;

using namespace cv;
using namespace std;


std::unique_ptr<ApiHandler> apiHandler;
std::queue<Mat> frameQueue;
std::queue<Mat> displayQueue;
std::mutex frameMutex;
std::mutex displayMutex;
std::condition_variable frameCV;
std::condition_variable displayCV;
std::atomic<bool> shouldExit(false);
const int MAX_QUEUE_SIZE = 10; // Maximum number of frames to keep in the queue keep low to reduce latency and memory usage!!!

// Movement event callback function for tracker when person event happens
void movementEventCallback(const TrackedPerson& person, const std::string& eventType) {
    if (apiHandler) {
        apiHandler->onPersonEvent(person, eventType);
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
        return loadConfig(settingsFile);
        LOG("Loaded config for device: " << config.deviceName);
    } catch (const std::exception& e) {
        ERROR("Error loading config: " << e.what());
        return Config();
    }
}

// This function is called by the capture thread to capture frames from the camera
void frameCapturingThread(VideoCapture& cap) {
    auto lastTime = chrono::steady_clock::now();
    auto lastPrintTime = chrono::steady_clock::now();

    while (!shouldExit) {
        Mat frame;
        cap.read(frame);

        auto currentTime = chrono::steady_clock::now();
        float elapsedMs = chrono::duration_cast<chrono::milliseconds>(currentTime - lastTime).count();
        float fps = (elapsedMs > 0) ? 1000.0f / elapsedMs : 0.0f;
        lastTime = currentTime;

        if (fps < 20.0f && fps > 0.0f) {
            auto now = chrono::steady_clock::now();
            if (chrono::duration_cast<chrono::seconds>(now - lastPrintTime).count() >= 1) {
                cout << "Warning: Capture FPS dropped below 20! Current FPS: " << fps 
                     << " | Queue Size: " << frameQueue.size() << endl;
                lastPrintTime = now;
            }
        }

        {
            std::unique_lock<std::mutex> lock(frameMutex);
            frameCV.wait(lock, [] { 
                return frameQueue.size() < MAX_QUEUE_SIZE || shouldExit; 
            });
            
            if (shouldExit) {
                frameCV.notify_all();
                break;
            }
            
            if (frameQueue.size() >= MAX_QUEUE_SIZE) {
                continue; // Drop frame
            }
            frameQueue.push(frame.clone());
        }
        
        frameCV.notify_one();
    }
}
// This function is called by the detection thread to process frames in the queue
void detectionThread(GenericDetector* detector, PeopleTracker& tracker) {
    while (!shouldExit) {
        Mat frame;
        {
            std::unique_lock<std::mutex> lock(frameMutex);
            frameCV.wait(lock, [] { return !frameQueue.empty() || shouldExit; });
            if (shouldExit) break;
            if (!frameQueue.empty()) {
                frame = std::move(frameQueue.front());
                frameQueue.pop();
            }
        }
        frameCV.notify_one();
        
        if (!frame.empty()) {
            detector->detect(frame);
            tracker.update(detector->getDetections(), frame.rows);
            tracker.draw(frame);
            {
                std::unique_lock<std::mutex> lock(displayMutex);
                displayCV.wait(lock, [] { 
                    return displayQueue.size() < MAX_QUEUE_SIZE || shouldExit; 
                });
                
                if (shouldExit) break;
                
                displayQueue.push(frame.clone());
            }
            
            displayCV.notify_one();
        }
    }
}

// This function is called by the display thread to show frames in the queue
void displayThread() {
    auto lastDetectionTime = chrono::steady_clock::now(); // Track last frame's detection time
    const int fpsBufferSize = 16;
    float fpsBuffer[fpsBufferSize] = {0.0};
    int frameCount = 0;
    
    while (!shouldExit) {
        Mat frame;
        
        {
            std::unique_lock<std::mutex> lock(displayMutex);
            displayCV.wait(lock, [] { 
                return !displayQueue.empty() || shouldExit; 
            });
            
            if (shouldExit) break;
            
            if (!displayQueue.empty()) {
                frame = displayQueue.front();
                displayQueue.pop();
            }
        }
        
        displayCV.notify_one();
        
        if (!frame.empty()) {
            // Calculate FPS based on time since last frame was processed by detection
            auto currentTime = chrono::steady_clock::now();
            float frameTimeMs = chrono::duration_cast<chrono::milliseconds>(currentTime - lastDetectionTime).count();
            lastDetectionTime = currentTime; // Update to current frame's dequeue time
            
            // Calculate FPS, cap at realistic max (e.g., camera's 30 FPS)
            float fps = (frameTimeMs > 0) ? std::min(1000.0f / frameTimeMs, 30.0f) : 0.0f;
            fpsBuffer[frameCount % fpsBufferSize] = fps;
            frameCount++;
            
            // Compute average FPS
            float avgFps = 0.0;
            for (int i = 0; i < min(frameCount, fpsBufferSize); i++) {
                avgFps += fpsBuffer[i];
            }
            avgFps /= min(frameCount, fpsBufferSize);
            
            string fpsText = format("FPS: %.2f", avgFps);
            putText(frame, fpsText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            
            imshow("People Detection and Tracking", frame);
            
            if (waitKey(1) == 'q') {
                shouldExit = true;
                frameCV.notify_all();
                displayCV.notify_all();
                break;
            }
        }
    }
    destroyAllWindows();
}

int main(int argc, char** argv) {
    config = loadSettings("../settings.json");
    PeopleTracker tracker;
    tracker.setMovementCallback(movementEventCallback);
    tracker.setEntranceZones(config.entranceZones);

    apiHandler.reset(new ApiHandler(config.serverEventAPI));
    
    GenericDetector* detector = createDetector(config.modelEnginePath, {"person"});
    if (!detector) {
        cerr << "Error: Failed to initialize detector." << endl;
        return -1;
    }
    cv::VideoCapture cap;
    // fix this pipeline no 30fps at 1280x720
    std::string pipeline = "v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink";
    cap.open(pipeline, cv::CAP_GSTREAMER);
    //cap.open(0, cv::CAP_V4L2); // Open camera with V4L2 backend
    //if (cap.isOpened()) {
    //    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    //    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    //    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    //    cap.set(cv::CAP_PROP_FPS, 30);
    //    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1); // 1=manual (V4L2 maps 1 to manual, 3 to auto)
    //} else {
    //    cerr << "Error: Could not open video capture device 0." << endl;
    //    return -1;
    //}
    cout << "Camera opened successfully. Resolution: " 
        << cap.get(CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CAP_PROP_FRAME_HEIGHT) << " framerate: " << cap.get(CAP_PROP_FPS) << endl;

    
    cout << "Press 'q' to quit." << endl;

    std::thread captureThread(frameCapturingThread, std::ref(cap));
    std::thread processingThread(detectionThread, detector, std::ref(tracker));
    std::thread showThread(displayThread);
    
    captureThread.join();
    processingThread.join();
    showThread.join();

    cap.release();
    destroyAllWindows();
    apiHandler.reset();
    delete detector;
    
    return 0;
}