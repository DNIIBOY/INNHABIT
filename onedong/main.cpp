#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <fstream>
#include <curl/curl.h>
#include <string>
#include <chrono>
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
};

struct BoxSize{
    int width;
    int height;
};


class Person {
    public:
        Position pos;
        BoxSize size;
        vector<Position> history;
        Scalar color;
        int killCount = 0;
        bool fromTop = false;
        bool fromBottom = false;

        Person() {};

        Person(Position pos, BoxSize size) {
            this->update(pos, size);
            this->color = Scalar(rand() % 255, rand() % 255, rand() % 255);
            int bottomY = pos.y + size.height / 2;
            int topY = pos.y - size.height / 2;
            this->fromTop = topY < 50;
            this->fromBottom = bottomY > 440;
        };
        void update(Position pos, BoxSize size) {
            this->pos = pos;
            this->size = size;
            this->history.push_back(pos);
            this->killCount = 0;
        };
        Rect getBoundingBox() {
            return Rect(
                pos.x - size.width / 2,
                pos.y - size.height / 2,
                size.width,
                size.height
            );
        };
        bool operator==(const Person& other) const {
            return this->pos.x == other.pos.x && this->pos.y == other.pos.y;
        };
};

class PeopleTracker {
    public:
        vector<Person> people;
        void update(vector<Person> newPeople) {
            for (Person& newPerson : newPeople) {
                Position pos = newPerson.pos;
                BoxSize size = newPerson.size;
                Person closestPerson;
                int minDistance = 999999;
                int closestPersonIndex = -1;
                for (int i = 0; i < people.size(); i++) {
                    Person person = people[i];
                    int distance = sqrt(pow(person.pos.x - pos.x, 2) + pow(person.pos.y - pos.y, 2));
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestPerson = person;
                        closestPersonIndex = i;
                    }
                }
                if (minDistance < 120) {
                    closestPerson.update(pos, size);
                    newPerson=closestPerson;
                    people.erase(people.begin() + closestPersonIndex);
                } else {
                    people.push_back(newPerson);
                }
            }
            for (Person& missingPerson : people) {
                missingPerson.killCount++;
                if (missingPerson.killCount < 10) {
                    newPeople.push_back(missingPerson);
                } else {
                    triggerMove(missingPerson);
                }
            }
            people = newPeople;
        };
        void draw(Mat frame) {
            for (Person person : people) {
                if (person.fromTop) {
                    putText(frame, "Top", Point(person.pos.x, person.pos.y), FONT_HERSHEY_SIMPLEX, 0.5, person.color, 2);
                }
                if (person.fromBottom) {
                    putText(frame, "Bottom", Point(person.pos.x, person.pos.y), FONT_HERSHEY_SIMPLEX, 0.5, person.color, 2);
                }
                Rect box = person.getBoundingBox();
                rectangle(frame, box, person.color, 2);
                for (Position pos : person.history) {
                    circle(frame, Point(pos.x, pos.y), 2, person.color, -1);
                }
            }
        };
        void triggerMove(Person person) {
            int bottomY = person.pos.y + person.size.height / 2;
            int topY = person.pos.y - person.size.height / 2;
            string jsonPayload = R"({"person": 2})";
            string serverUrl;
            if (person.fromTop && bottomY > 440) {
                serverUrl = "http://localhost:8000/exit";
                sendHttpRequest(serverUrl, jsonPayload);
                cout << "Person moved from top to bottom" << endl;
            }
            if (person.fromBottom && topY < 50) {
                serverUrl = "http://localhost:8000/enter";
                sendHttpRequest(serverUrl, jsonPayload);
                cout << "Person moved from bottom to top" << endl;
            }
        };
};
int main() {
    // Load YOLO model
    Net net = readNet("yolov7-tiny.weights", "yolov7-tiny.cfg");

    // Set the backend and target
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        cout << "Using CUDA backend" << endl;
    }
    else if (cv::ocl::haveOpenCL()) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
        cout << "Using OpenCL backend" << endl;
    }
    else {
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
    VideoCapture cap(0, CAP_V4L2);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 320);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open webcam" << endl;
        return -1;
    }

    // Tracker initialization
    vector<Ptr<TrackerCSRT>> trackers;
    vector<Rect> bboxes;
    bool trackingInitialized = false;

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        int width = frame.cols;
        int height = frame.rows;

        // Convert the frame to a blob (required by YOLO)
        Mat blob;
        blobFromImage(frame, blob, 0.00392, Size(320, 320), Scalar(0, 0, 0), true, false);
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
                    boxes.emplace_back(centerX - w / 2, centerY - h / 2, w, h);
                    confidences.push_back(confidence);
                    classIds.push_back(classId);
                }
            }
        }

        // Non-maxima suppression
        vector<int> indices;
        NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        // Initialize tracker for each detected person in the first frame
        if (!trackingInitialized && !indices.empty()) {
            for (int i : indices) {
                Rect personBox = boxes[i];
                bboxes.push_back(Rect2d(personBox.x, personBox.y, personBox.width, personBox.height));
                Ptr<TrackerCSRT> tracker = TrackerCSRT::create();
                tracker->init(frame, bboxes.back());
                trackers.push_back(tracker);
            }
            trackingInitialized = true;
        }

        // Update all trackers
        if (trackingInitialized) {
            for (size_t i = 0; i < trackers.size(); i++) {
                bool success = trackers[i]->update(frame, bboxes[i]);
                if (success) {
                    // Draw bounding box for each tracked person
                    rectangle(frame, bboxes[i], Scalar(0, 255, 0), 2);
                    putText(frame, "Perso " + to_string(i + 1), Point(bboxes[i].x, bboxes[i].y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
                }
                else {
                    // If tracking fails, you can reset the tracker
                    putText(frame, "Tracking failure", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
    
    // Print the elapsed time in seconds
        std::cout << "Time taken: " << duration.count() << " seconds\n";
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
