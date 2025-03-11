#include "detector.h"
#include "postprocess.h"
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class CPUDetector : public GenericDetector {
private:
    Net net;
    vector<string> classes;
    
    // Changed from static constexpr to static const
    static const char* const labels[80];

public:
    CPUDetector(const string& modelPath, const vector<string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_) {
        initialize(modelPath);
    }

    void detect(Mat& frame) override {
        if (!initialized) {
            cerr << "Error: Detector not properly initialized" << endl;
            return;
        }

        Mat img;
        cvtColor(frame, img, COLOR_BGR2RGB);
        int img_width = img.cols;
        int img_height = img.rows;

        Mat resized_img(height, width, CV_8UC3, Scalar(114, 114, 114));
        float scale = min(static_cast<float>(width) / img_width, static_cast<float>(height) / img_height);
        int new_width = static_cast<int>(img_width * scale);
        int new_height = static_cast<int>(img_height * scale);
        int dx = (width - new_width) / 2;
        int dy = (height - new_height) / 2;

        Mat resized_part;
        resize(img, resized_part, Size(new_width, new_height));
        resized_part.copyTo(resized_img(Rect(dx, dy, new_width, new_height)));

        // Direct OpenCV DNN processing for CPU
        Mat blob = blobFromImage(resized_img, 1/255.0, Size(width, height), Scalar(0,0,0), true, false);
        net.setInput(blob);
        
        vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());
        
        // Process detections directly using OpenCV instead of the post_process function
        vector<Rect> boxes;
        vector<float> confidences;
        vector<int> classIds;
        
        // Process outputs
        for (size_t i = 0; i < outs.size(); ++i) {
            // For YOLOv7, we need to process the detection outputs
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j) {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                // Get the value and location of the maximum score
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                
                if (confidence > BOX_THRESH) {
                    int centerX = (int)(data[j * outs[i].cols + 0] * frame.cols);
                    int centerY = (int)(data[j * outs[i].cols + 1] * frame.rows);
                    int width = (int)(data[j * outs[i].cols + 2] * frame.cols);
                    int height = (int)(data[j * outs[i].cols + 3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
        
        // Apply non-maximum suppression
        vector<int> indices;
        NMSBoxes(boxes, confidences, BOX_THRESH, NMS_THRESH, indices);
                
        // Draw bounding boxes and labels
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Rect box = boxes[idx];
            int classId = classIds[idx];
            
            // Filter by target classes if specified
            if (!targetClasses.empty()) {
                bool isTarget = false;
                for (const auto& target : targetClasses) {
                    // Use classId as index to get class name from static labels array
                    const char* className = classId < 80 ? labels[classId] : "unknown";
                    if (target == className) {
                        isTarget = true;
                        break;
                    }
                }
                if (!isTarget) continue;
            }
            
            // Draw bounding box
            rectangle(frame, box, Scalar(0, 255, 0), 2);
            
            // Get class name and confidence
            string className = (classId < 80) ? labels[classId] : "unknown";
            string label = className + ": " + to_string(int(confidences[idx] * 100)) + "%";
            
            // Draw label background
            int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            rectangle(frame, Point(box.x, box.y - labelSize.height - baseLine),
                      Point(box.x + labelSize.width, box.y), Scalar(255, 255, 255), FILLED);
            
            // Draw label text
            putText(frame, label, Point(box.x, box.y - baseLine), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }
        
    }

protected:
    void initialize(const string& modelPath) override {
        string cfg = modelPath + "/yolov7-tiny.cfg";
        string weights = modelPath + "/yolov7-tiny.weights";
        
        // Load COCO class names
        string namesFile = modelPath + "/coco.names";
        ifstream ifs(namesFile);
        if (!ifs.is_open()) {
            cerr << "Error opening names file: " << namesFile << endl;
            throw runtime_error("Failed to load class names");
        }
        string line;
        while (getline(ifs, line)) {
            classes.push_back(line);
        }
        
        net = readNet(cfg, weights);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

        if (net.empty()) {
            throw runtime_error("Failed to load YOLOv  model for CPU");
        }

        width = 640;  // Hardcoded for simplicity, adjust as needed
        height = 640;
        channel = 3;
        initialized = true;
    }

    DetectionOutput runInference(const Mat& input) override {
        // This is not used in the overridden detect() method but must be implemented for the abstract class
        DetectionOutput output;
        output.num_outputs = 0;
        return output;
    }

    ~CPUDetector() override {

    }
};

// Define the static array outside the class
const char* const CPUDetector::labels[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

Detector* createDetector(const string& modelPath, const vector<string>& targetClasses) {
    try {
#ifdef DEBUG
        cout << "Creating CPUDetector..." << endl;
#endif
        return new CPUDetector(modelPath, targetClasses);
    } catch (const exception& e) {
        cerr << "Error creating CPU detector: " << e.what() << endl;
        return nullptr;
    }
}