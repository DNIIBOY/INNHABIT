#include "opencv2/videoio.hpp"
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <fstream>
#include <curl/curl.h>
#include <string>

using namespace cv;
using namespace dnn;
using namespace std;

void sendHttpRequest(const string& url, const string& jsonPayload) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        cerr << "Failed to initialize CURL" << endl;
        return;
    }

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonPayload.c_str());

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        cerr << "CURL request failed: " << curl_easy_strerror(res) << endl;
    }

    curl_easy_cleanup(curl);
}



struct Position{
    int x;
    int y;

    Position operator+(const Position& other) const {
        return {x + other.x, y + other.y};
    }

    Position operator-(const Position& other) const {
        return {x - other.x, y - other.y};
    }

    static Position average(const Position& p1, const Position& p2) {
        return { (p1.x + p2.x) / 2, (p1.y + p2.y) / 2 };
    }

    int magnitude() const {
        return std::sqrt(x * x + y * y);
    }

    Position normalize() const {
        int magnitude = std::sqrt(x * x + y * y);
        if (magnitude == 0) return {0, 0}; // Avoid division by zero
        return {x / magnitude, y / magnitude};
    }

    Position multiplyByScalar(int scalar) {
        return {x * scalar, y * scalar};
    }
    
};

struct BoxSize{
    int width;
    int height;
};

struct Detection {
    int id;
    Rect bbox;
    float confidence;
    bool matched = false;
    Scalar color;
    int killCount = 0;
};

float computeIoU(const Rect& box1, const Rect& box2) {
    float intersection = (box1 & box2).area();
    float unionArea = box1.area() + box2.area() - intersection;
    return intersection / unionArea;
}


class ByteTrack {
    private:
        vector<Detection> activeTracks;
        int nextID = 1;
        int maxKillCount = 10;
        float iouThreshold = 0.3;
        float confThresholdHigh = 0.5; // Threshold for high-confidence detections
        float confThresholdLow = 0.3;  // Threshold for low-confidence detections
    
    public:
        vector<Detection> update(vector<Detection>& detections) {
            vector<Detection> highConfDetections;
            vector<Detection> lowConfDetections;
            vector<Detection> unmatchedTracks;
    
            // Separate high and low-confidence detections
            for (auto& det : detections) {
                if (det.confidence > confThresholdHigh) {
                    highConfDetections.push_back(det);
                } else if (det.confidence > confThresholdLow) {
                    lowConfDetections.push_back(det);
                }
            }
    
            // Step 1: Match high-confidence detections to existing tracks
            for (auto& track : activeTracks) {
                track.matched = false; // Set all of the classes tracks to be unmatched
                for (auto& det : highConfDetections) {
                    if (!det.matched && computeIoU(track.bbox, det.bbox) > iouThreshold) {
                        track.killCount = 0;
                        track.bbox = det.bbox;
                        track.confidence = det.confidence;
                        track.matched = true;
                        det.matched = true;
                        break;
                    }
                }
                if (!track.matched) {
                    // Step 2: Assign remaining unmatched tracks to low-confidence detections
                    for (auto& det : lowConfDetections) {
                        if (!det.matched && computeIoU(track.bbox, det.bbox) > iouThreshold) {
                            track.killCount = 0;
                            track.bbox = det.bbox;
                            track.confidence = det.confidence;
                            //track.color = det.color;
                            track.matched = true;
                            det.matched = true;
                            break;
                        }
                    }
                }
                
            }
    
            // Step 3:
            for (auto it = activeTracks.begin(); it != activeTracks.end(); ) {
                auto& track = *it;
            
                if (!track.matched) {
                    track.killCount++;
                    
                    // If the track has been unmatched for too long, remove it
                    if (track.killCount > maxKillCount) {
                        // Track should be removed from activeTracks, so we erase it
                        it = activeTracks.erase(it);  // erase returns the next iterator
                    } else {
                        // If not removed, just move to the next track
                        ++it;
                    }
                } else {
                    // If track is matched, move to the next track
                    ++it;
                }
            }
            
    
            // Step 4: Create new tracks for unmatched high-confidence detections
            for (auto& det : highConfDetections) {
                if (!det.matched) {
                    det.id = nextID++;
                    det.color = Scalar(rand() % 255, rand() % 255, rand() % 255);
                    activeTracks.push_back(det);
                }
            }
    
            return activeTracks;
        }

    };



int main(int argc, char* argv[]) {
    std::string imagePath = "WIN_20250303_10_21_48_Pro.mp4";
    if (argc > 1) {
        imagePath = argv[1];
        std::cout << "Received image path: " << imagePath << '\n';
    }

    // Load YOLO model
    Net net = readNet("yolov7-tiny.weights", "yolov7-tiny.cfg");

    // Set the backend and target
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        // If CUDA is available, use it
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        cout << "Using CUDA backend" << endl;
    } 
    else if (cv::ocl::haveOpenCL()) {
        // If OpenCL is available, use it
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
        cout << "Using OpenCL backend" << endl;
    } 
    else {
        // Fall back to CPU if neither CUDA nor OpenCL is available
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        cout << "Using CPU backend" << endl;
    }

    vector<string> layerNames = net.getUnconnectedOutLayersNames();

    // Load class labels (COCO dataset labels)
    vector<string> classes;
    ifstream ifs("coco.names");
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }

    // Open the laptop camera (use 0 for default webcam)
    VideoCapture cap(imagePath, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open webcam" << endl;
        return -1;
    }

    ByteTrack tracker;

    
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        vector<Detection> detections;

        // Prepare input blob for YOLOv7
        Mat blob;
        blobFromImage(frame, blob, 0.00392, Size(320, 320), Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        // Perform forward pass and get output layers
        vector<Mat> outputs;
        net.forward(outputs, layerNames);

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        // Process the outputs
        for (const auto& output : outputs) {
            for (int i = 0; i < output.rows; i++) {
                const float* data = output.ptr<float>(i);
                vector<float> scores(data + 5, data + output.cols);
                int classId = max_element(scores.begin(), scores.end()) - scores.begin();
                float confidence = scores[classId];; // Confidence score for each detected object
    
                if (confidence > 0.1 && classId == 0) {
                    // The data format depends on YOLO model
                    // YOLOv7 has 4 values for bbox (x_center, y_center, width, height) followed by class scores
    
                    int centerX = static_cast<int>(data[0] * frame.cols);
                    int centerY = static_cast<int>(data[1] * frame.rows);
                    int width = static_cast<int>(data[2] * frame.cols);
                    int height = static_cast<int>(data[3] * frame.rows);
    
                    // Create a Rect bounding box around the detected object
                    int x = centerX - width / 2;
                    int y = centerY - height / 2;
                    Rect box(x, y, width, height);
    
                    // Push detected class IDs, confidence, and bounding box coordinates
                    confidences.push_back(confidence);
                    boxes.push_back(box);
                    classIds.push_back(0);  // Assuming we're detecting "people" (class 0 in COCO dataset)
                }
            }
        }

        // Apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes
        vector<int> indices;
        dnn::NMSBoxes(boxes, confidences, 0.2, 0.2, indices);

        // Store final detections with unique IDs
        for (int i : indices) {
            Detection detection;
            detection.bbox = boxes[i];
            detection.confidence = confidences[i];
            detection.id = -1;  // ID will be assigned later during tracking
            detections.push_back(detection);
        }

        vector<Detection> trackedObjects = tracker.update(detections);

        for (const auto& obj : trackedObjects) {
            
            rectangle(frame, obj.bbox, obj.color, 2);
            putText(frame, "ID: " + to_string(obj.id), obj.bbox.tl(), FONT_HERSHEY_SIMPLEX, 0.5, obj.color, 2);
        }

        // Draw bounding boxes for detected humans
        /*vector<Person> people;
        for (int i : indices) {
            Rect box = boxes[i];
            Person person = Person({box.x, box.y}, {box.width, box.height});
            people.push_back(person);
        }
        tracker.update(people);
        tracker.draw(frame);*/

        // Show the output frame
        imshow("Human Detection", frame);

        // Break on 'q' key press
        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
