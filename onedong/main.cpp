#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace dnn;
using namespace std;

int main() {
    // Load YOLO model
    Net net = readNet("yolov3.weights", "yolov3.cfg");
    vector<string> layerNames = net.getUnconnectedOutLayersNames();

    // Load class labels (COCO dataset labels)
    vector<string> classes;
    ifstream ifs("coco.names");
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }

    // Open the laptop camera (use 0 for default webcam)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open webcam" << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        int width = frame.cols;
        int height = frame.rows;

        // Convert the frame to a blob (required by YOLO)
        Mat blob;
        blobFromImage(frame, blob, 0.00392, Size(416, 416), Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        vector<Mat> outs;
        net.forward(outs, layerNames);

        // Post-process the detections
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        for (auto& output : outs) {
            for (int i = 0; i < output.rows; i++) {
                float* data = output.ptr<float>(i);
                vector<float> scores(data + 5, data + output.cols);
                int classId = max_element(scores.begin(), scores.end()) - scores.begin();
                float confidence = scores[classId];
                if (confidence > 0.5 && classes[classId] == "person") {
                    int centerX = static_cast<int>(data[0] * width);
                    int centerY = static_cast<int>(data[1] * height);
                    int w = static_cast<int>(data[2] * width);
                    int h = static_cast<int>(data[3] * height);
                    int x = centerX - w / 2;
                    int y = centerY - h / 2;
                    boxes.emplace_back(x, y, w, h);
                    confidences.push_back(confidence);
                    classIds.push_back(classId);
                }
            }
        }

        // Non-maxima suppression
        vector<int> indices;
        NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        // Draw bounding boxes for detected humans
        for (int i : indices) {
            Rect box = boxes[i];
            string label = classes[classIds[i]] + " " + to_string(confidences[i]);
            rectangle(frame, box, Scalar(0, 255, 0), 2);
            putText(frame, label, Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        }

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
