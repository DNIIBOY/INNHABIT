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
const int MAX_QUEUE_SIZE = 5; // Maximum number of frames to keep in the queue keep low to reduce latency and memory usage!!!

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

Config loadSettings(const string& settingsFile) {
    try {
        return loadConfig(settingsFile);
        LOG("Loaded config for device: " << config.deviceName);
    } catch (const std::exception& e) {
        ERROR("Error loading config: " << e.what());
        return Config();
    }
}

void frameCapturingThread(VideoCapture& cap) {
    auto startTime = chrono::steady_clock::now();
    int frameCount = 0;
    while (!shouldExit) {
        Mat frame;
        cap >> frame;
        
        if (frame.empty()) {
            cerr << "Error: Frame capture failed in capture thread." << endl;
            shouldExit = true;
            break;
        }
        
        frameCount++;
        auto endTime = chrono::steady_clock::now();
        float elapsedMs = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
        if (elapsedMs >= 1000.0f) {
            float captureFps = (elapsedMs > 0) ? 1000.0f * frameCount / elapsedMs : 0.0f;
            cout << "Capture FPS: " << captureFps << endl;
            frameCount = 0;
            startTime = endTime;
        }
        
        {
            std::unique_lock<std::mutex> lock(frameMutex);
            frameCV.wait(lock, [] { 
                return frameQueue.size() < MAX_QUEUE_SIZE || shouldExit; 
            });
            
            if (shouldExit) break;
            
            frameQueue.push(frame.clone());
        }
        
        frameCV.notify_one();
    }
}

void detectionThread(GenericDetector* detector, PeopleTracker& tracker) {
    while (!shouldExit) {
        Mat frame;
        
        {
            std::unique_lock<std::mutex> lock(frameMutex);
            frameCV.wait(lock, [] { 
                return !frameQueue.empty() || shouldExit; 
            });
            
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

void displayThread() {
    auto startTime = chrono::steady_clock::now();
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
            auto endTime = chrono::steady_clock::now();
            float frameTimeMs = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
            startTime = endTime;
            
            float fps = (frameTimeMs > 0) ? 1000.0f / frameTimeMs : 0.0f;
            fpsBuffer[frameCount % fpsBufferSize] = fps;
            frameCount++;
            
            float avgFps = 0.0;
            for (int i = 0; i < min(frameCount, fpsBufferSize); i++) {
                avgFps += fpsBuffer[i];
            }
            avgFps /= min(frameCount, fpsBufferSize);
            
            string fpsText = format("Display FPS: %.2f", avgFps);
            putText(frame, fpsText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            
            Mat displayFrame;
            resize(frame, displayFrame, Size(640, 360));
            imshow("People Detection and Tracking", displayFrame);
            
            if (waitKey(1) == 'q') {
                shouldExit = true;
                break;
            }
        }
    }
}

int main(int argc, char** argv) {
    string videoFile;
    string modelPath;
    config = loadSettings("../settings.json");
    
    string apiUrl = config.ServerEventAPI;
    
    PeopleTracker tracker;
    tracker.setMovementCallback(movementEventCallback);
    tracker.setEntranceZones(config.entranceZones);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--video") == 0 && i + 1 < argc) {
            videoFile = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            modelPath = argv[++i];
        } else {
            cerr << "Unknown argument: " << argv[i] << endl;
            printUsage(argv[0]);
            return -1;
        }
    }

    apiHandler.reset(new ApiHandler(apiUrl));
    
    GenericDetector* detector = createDetector(modelPath, {"person"});
    if (!detector) {
        cerr << "Error: Failed to initialize detector." << endl;
        return -1;
    }
    
    cv::VideoCapture cap;

    if (!videoFile.empty()) {
        cap.open(videoFile);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open video file: " << videoFile << endl;
            return -1;
        }
        cout << "Video file opened successfully. Resolution: " 
             << cap.get(CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
    } else {
        std::string pipeline = "v4l2src device=/dev/video0 ! image/jpeg, width=(int)1280, height=(int)720, framerate=(fraction)30/1 ! jpegdec ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1 max-buffers=2";
        cap.open(pipeline, cv::CAP_GSTREAMER);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open video capture device 0." << endl;
            return -1;
        }
        cout << "Camera opened successfully. Resolution: " 
             << cap.get(CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CAP_PROP_FRAME_HEIGHT) << " framerate: " << cap.get(CAP_PROP_FPS) << endl;
    }
    
    cout << "Press 'q' to quit." << endl;

    std::thread captureThread(frameCapturingThread, std::ref(cap));
    std::thread processingThread(detectionThread, detector, std::ref(tracker));
    std::thread visThread(displayThread);
    
    captureThread.join();
    processingThread.join();
    visThread.join();

    cap.release();
    destroyAllWindows();
    apiHandler.reset();
    delete detector;
    
    return 0;
}