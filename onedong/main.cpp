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

Position averageVelocity(const std::vector<Position>& history, int frames) {
    int size = history.size();
    if (size < 2) return {0, 0}; // Need at least 2 positions to compute velocity

    int count = std::min(size - 1, frames); // Up to last "frames" intervals
    Position sumVelocity = {0, 0};

    // Sum up velocity differences over last `count` intervals
    for (int i = size - count; i < size - 1; ++i) {
        sumVelocity = sumVelocity + (history[i + 1] - history[i]);
    }

    // Compute the average velocity
    return {sumVelocity.x / count, sumVelocity.y / count};
}

struct BoxSize{
    int width;
    int height;
};


class Person {
    public:
        Position pos;
        BoxSize size;
        vector<Position> history;
        int recentVelocity;
        Position directionVector; // The average velocity over a longer period of time
        Position velocity;
        Scalar color;
        int killCount = 0;
        bool fromTop = false;
        bool fromBottom = false;

        Position expectedPos;

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
            this->expectedPos = pos;
            this->size = size;
            this->history.push_back(pos);
            this->killCount = 0;
            this->recentVelocity = averageVelocity(this->history,5).magnitude();
            this->directionVector = averageVelocity(this->history,20).normalize();
            this->velocity = directionVector.multiplyByScalar(recentVelocity);
            
        };
        void missingUpdate() {
            this->killCount += 1;
            this->expectedPos = expectedPos + velocity;
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
        vector<Person> peopleManifest;
        void update(vector<Person> peopleDetectedThisFrame) {
            for (Person& detectedPerson : peopleDetectedThisFrame) {
                Position pos = detectedPerson.pos;
                BoxSize size = detectedPerson.size;
                Person closestPerson;
                int minDistance = 999999;
                int closestPersonIndex = -1;
                // Find nearest already existing person
                for (int i = 0; i < peopleManifest.size(); i++) {
                    Person person = peopleManifest[i];
                    int distance = sqrt(pow(person.pos.x - pos.x, 2) + pow(person.pos.y - pos.y, 2));
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestPerson = person;
                        closestPersonIndex = i;
                    }
                }
                // If the closest one is within 120 px, this must be the same person as from last frame
                if (minDistance < 120) {
                    closestPerson.update(pos, size); // Set the closestPerson pos and size, so they match the new one
                    detectedPerson=closestPerson; // Give this detected person the values of the closest person
                    peopleManifest.erase(peopleManifest.begin() + closestPersonIndex); // Delete this closestPerson, so it is not treated as a missing person
                } /*else {
                    peopleManifest.push_back(detectedPerson); // Add
                }*/
            }
            for (Person& missingPerson : peopleManifest) {
                missingPerson.missingUpdate();
                // if the character has not been missing for too long, keep it alive
                if (missingPerson.killCount < 30) {
                    peopleDetectedThisFrame.push_back(missingPerson);
                } else { // If the person has been gone for too long, it is told to move
                    triggerMove(missingPerson);
                }
            }
            peopleManifest = peopleDetectedThisFrame;
        };
        void draw(Mat frame) {
            
            for (Person person : peopleManifest) {
                if (person.killCount == 0) {
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
                
                // Draw velocity vector
                Point start(person.pos.x, person.pos.y); 
                Point end(person.pos.x + person.velocity.x, person.pos.y + person.velocity.y);
                arrowedLine(frame, start, end, person.color, 2, LINE_AA, 0, 10.0); // 0.2 for arrow scale
                
                if (person.killCount > 0) {

                    putText(frame, "(Missing)", Point(person.pos.x+20, person.pos.y), FONT_HERSHEY_SIMPLEX, 0.5, person.color, 2);


                    // Draw line from last known position to expected position
                    line(frame, start, Point(person.expectedPos.x, person.expectedPos.y), 5);

                    // Draw the expected position
                    circle(frame, Point(person.expectedPos.x, person.expectedPos.y), 3, (255,0,0), 2);

                    // Draw the space in which a person is looked at
                    Position averageOfExpectedPosAndPos = Position::average(person.expectedPos, person.pos);
                    circle(frame, Point(averageOfExpectedPosAndPos.x, averageOfExpectedPosAndPos.y), person.killCount, person.color, 2);
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

int main(int argc, char* argv[]) {
    std::string imagePath = "WIN_20250303_10_21_48_Pro.mp4";
    if (argc > 1) {
        imagePath = argv[1];
        std::cout << "Received image path: " << imagePath << '\n';
    }

    // Load YOLO model
    Net net = readNet("yolov7-tiny.weights", "yolov7-tiny.cfg");
    PeopleTracker tracker;
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

    while (true) {
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
                    boxes.emplace_back(centerX, centerY, w, h);
                    confidences.push_back(confidence);
                    classIds.push_back(classId);
                }
            }
        }

        // Non-maxima suppression
        vector<int> indices;
        NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        // Draw bounding boxes for detected humans
        vector<Person> people;
        for (int i : indices) {
            Rect box = boxes[i];
            Person person = Person({box.x, box.y}, {box.width, box.height});
            people.push_back(person);
        }
        tracker.update(people);
        tracker.draw(frame);

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
