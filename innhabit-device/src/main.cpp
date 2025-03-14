#include "api_handler.h"
#include "detector.h"
#include "tracker.h"
#include "common.h"

Config config;
using namespace cv;
using namespace std;

std::unique_ptr<ApiHandler> apiHandler;

void movementEventCallback(const TrackedPerson& person, const std::string& eventType) {
    if (apiHandler) {
        apiHandler->onPersonEvent(person, eventType);
        // Optionally log the specific zone event
        cout << "Person " << person.id << " " << eventType << endl;
    }
}

void printUsage(const char* progName) {
    cout << "Usage: " << progName << " [--video <video_file>] [--image <image_file>]" << endl;
    cout << "  --video <file> : Process a video file" << endl;
    cout << "  --image <file> : Process a single image" << endl;
    cout << "  --api <url>    : API server URL (default: http://127.0.0.1:8000)" << endl;
    cout << "  (No arguments defaults to webcam)" << endl;
}

int main(int argc, char** argv) {
    string modelPath = "../models";
    string videoFile;
    string imageFile;
    
    try {
        config = loadConfig("../settings.json");
        LOG("Loaded config for device: " << config.deviceName);
    } catch (const std::exception& e) {
        ERROR("Error loading config: " << e.what());
        return -1;
    }
    
    string apiUrl = "http://" + config.serverIP + ":" + config.serverPort;
    cout << "API URL: " << apiUrl << endl;
    
    PeopleTracker tracker;
    tracker.setMovementCallback(movementEventCallback);
    tracker.setEntranceZones(config.entranceZones);  // Pass vector of zones

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--video") == 0 && i + 1 < argc) {
            videoFile = argv[++i];
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            imageFile = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (strcmp(argv[i], "--api") == 0 && i + 1 < argc) {
            apiUrl = argv[++i];
        } else {
            cerr << "Unknown argument: " << argv[i] << endl;
            printUsage(argv[0]);
            return -1;
        }
    }

    if (!videoFile.empty() && !imageFile.empty()) {
        cerr << "Error: Cannot specify both --video and --image" << endl;
        printUsage(argv[0]);
        return -1;
    }
    
    vector<string> targetClasses = {"person"};

    try {
        apiHandler.reset(new ApiHandler(apiUrl));
        cout << "API handler initialized with URL: " << apiUrl << endl;
    } catch (const std::exception& e) {
        cerr << "Error initializing API handler: " << e.what() << endl;
        cerr << "The application will continue without API connectivity." << endl;
    }

    unique_ptr<Detector> detector(createDetector(modelPath, targetClasses));
    if (!detector) {
        cerr << "Error: Failed to initialize detector." << endl;
        return -1;
    }
    
    Mat frame;
    VideoCapture cap;

    if (!videoFile.empty()) {
        cap.open(videoFile);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open video file: " << videoFile << endl;
            return -1;
        }
        cout << "Video file opened successfully. Resolution: " 
             << cap.get(CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
    } else if (!imageFile.empty()) {
        frame = imread(imageFile);
        if (frame.empty()) {
            cerr << "Error: Could not open image file: " << imageFile << endl;
            return -1;
        }
        cout << "Image file opened successfully. Resolution: " 
             << frame.cols << "x" << frame.rows << endl;
    } else {
        cap.open(0);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open video capture device 0." << endl;
            return -1;
        }
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Initial frame capture failed." << endl;
            cap.release();
            return -1;
        }
        cout << "Camera opened successfully. Resolution: " 
             << frame.cols << "x" << frame.rows << endl;
    }

    const int fpsBufferSize = 16;
    float fpsBuffer[fpsBufferSize] = {0.0};
    int frameCount = 0;
    chrono::steady_clock::time_point startTime;

    if (!imageFile.empty()) {
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
        waitKey(0);
    } else {
        cout << "Press 'q' to quit." << endl;
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
    apiHandler.reset();
    
    return 0;
}