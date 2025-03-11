#include "detector.h"
#include "tracker.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include <curl/curl.h>
#include <cstring>  // Added for strcmp

using namespace cv;
using namespace std;

// Global curl handle
CURL* curlHandle = nullptr;

// Callback for person movements
void onPersonMovement(const TrackedPerson& person, const string& direction) {
    if (!curlHandle) return;
    
    string jsonPayload = "{\"person\": " + to_string(person.id) + "}";
    string url = "http://localhost:8000/" + direction;
    
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curlHandle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curlHandle, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curlHandle, CURLOPT_POST, 1L);
    curl_easy_setopt(curlHandle, CURLOPT_POSTFIELDS, jsonPayload.c_str());

    CURLcode res = curl_easy_perform(curlHandle);
    if (res != CURLE_OK) {
        cerr << "CURL request failed: " << curl_easy_strerror(res) << endl;
    }

    curl_slist_free_all(headers);
    
    cout << "Person ID " << person.id << " " << direction << " event sent" << endl;
}

vector<string> loadClassNames(const string& filename) {
    vector<string> classNames;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open class names file: " << filename << endl;
        return classNames;
    }
    
    string line;
    while (getline(file, line)) {
        if (!line.empty()) classNames.push_back(line);
    }
    file.close();
    return classNames;
}

void printUsage(const char* progName) {
    cout << "Usage: " << progName << " [--video <video_file>] [--image <image_file>]" << endl;
    cout << "  --video <file> : Process a video file" << endl;
    cout << "  --image <file> : Process a single image" << endl;
    cout << "  (No arguments defaults to webcam)" << endl;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv) {
    // Initialize curl
    curl_global_init(CURL_GLOBAL_ALL);
    curlHandle = curl_easy_init();

    string classFile = "../models/coco.names";
    string modelPath = "../models";
    string videoFile;
    string imageFile;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--video") == 0 && i + 1 < argc) {
            videoFile = argv[++i];
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            imageFile = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            modelPath = argv[++i];
        } else {
            cerr << "Unknown argument: " << argv[i] << endl;
            cerr << "Usage: ./people_tracker [--video <video_file>] [--image <image_file>] [--model <model_path>]" << endl;
            return -1;
        }
    }

    classFile = modelPath + "/coco.names";

    // Validate arguments
    if (!videoFile.empty() && !imageFile.empty()) {
        cerr << "Error: Cannot specify both --video and --image" << endl;
        printUsage(argv[0]);
        if (curlHandle) curl_easy_cleanup(curlHandle);
        curl_global_cleanup();
        return -1;
    }

    vector<string> allClassNames = loadClassNames(classFile);
    if (allClassNames.empty()) {
        cerr << "Failed to load class names. Exiting." << endl;
        if (curlHandle) curl_easy_cleanup(curlHandle);
        curl_global_cleanup();
        return -1;
    }
    
    vector<string> targetClasses = {"person"};  // Track people by default

    unique_ptr<Detector> detector(createDetector(modelPath, targetClasses));
    if (!detector) {
        cerr << "Error: Failed to initialize detector." << endl;
        if (curlHandle) curl_easy_cleanup(curlHandle);
        curl_global_cleanup();
        return -1;
    }
    // Initialize tracker with reasonable parameters
    PeopleTracker tracker(10, 120.0f, 0.1f, 0.9f);
    tracker.setMovementCallback(onPersonMovement);
    
    Mat frame;
    VideoCapture cap;

    // Open input source based on arguments
    if (!videoFile.empty()) {
        cap.open(videoFile);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open video file: " << videoFile << endl;
            if (curlHandle) curl_easy_cleanup(curlHandle);
            curl_global_cleanup();
            return -1;
        }
        cout << "Video file opened successfully. Resolution: " 
             << cap.get(CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
    } else if (!imageFile.empty()) {
        frame = imread(imageFile);
        if (frame.empty()) {
            cerr << "Error: Could not open image file: " << imageFile << endl;
            if (curlHandle) curl_easy_cleanup(curlHandle);
            curl_global_cleanup();
            return -1;
        }
        cout << "Image file opened successfully. Resolution: " 
             << frame.cols << "x" << frame.rows << endl;
    } else {
        cap.open(0);  // Default to webcam
        if (!cap.isOpened()) {
            cerr << "Error: Could not open video capture device 0." << endl;
            cerr << "Check permissions (e.g., 'sudo chmod 666 /dev/video0')." << endl;
            if (curlHandle) curl_easy_cleanup(curlHandle);
            curl_global_cleanup();
            return -1;
        }
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Initial frame capture failed." << endl;
            cap.release();
            if (curlHandle) curl_easy_cleanup(curlHandle);
            curl_global_cleanup();
            return -1;
        }
        cout << "Camera opened successfully. Resolution: " 
             << frame.cols << "x" << frame.rows << endl;
    }

    const int fpsBufferSize = 16;
    float fpsBuffer[fpsBufferSize] = {0.0};
    int frameCount = 0;
    chrono::steady_clock::time_point startTime;

    // Main processing loop
    if (!imageFile.empty()) {
        // Single image processing
        startTime = chrono::steady_clock::now();

        detector->detect(frame);
        tracker.update(detector->getDetections(), frame.rows);
        tracker.draw(frame);

        auto endTime = chrono::steady_clock::now();
        float frameTimeMs = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
        float fps = (frameTimeMs > 0) ? 1000.0f / frameTimeMs : 0.0f;
        string fpsText = format("FPS: %.2f", fps);
        putText(frame, fpsText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

        imshow("People Detection and Tracking", frame);
        waitKey(0);  // Wait for any key press
    } else {
        // Video or webcam processing
        while (true) {
            startTime = chrono::steady_clock::now();
            
            cap >> frame;
            if (frame.empty()) {
                cerr << "Error: Frame capture failed during loop." << endl;
                break;
            }
            
            detector->detect(frame);
            tracker.update(detector->getDetections(), frame.rows);
            tracker.draw(frame);
            
            auto endTime = chrono::steady_clock::now();
            float frameTimeMs = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
            float fps = (frameTimeMs > 0) ? 1000.0f / frameTimeMs : 0.0f;
            
            fpsBuffer[frameCount % fpsBufferSize] = fps;
            frameCount++;
            
            float avgFps = 0.0;
            for (int i = 0; i < min(frameCount, fpsBufferSize); i++) {
                avgFps += fpsBuffer[i];
            }
            avgFps /= min(frameCount, fpsBufferSize);
            
            string fpsText = format("FPS: %.2f", avgFps);
            putText(frame, fpsText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            
            imshow("People Detection and Tracking", frame);
            if (waitKey(1) == 'q') break;
        }
    }
    
    cap.release();
    destroyAllWindows();
    
    // Cleanup curl
    if (curlHandle) curl_easy_cleanup(curlHandle);
    curl_global_cleanup();
    
    return 0;
}