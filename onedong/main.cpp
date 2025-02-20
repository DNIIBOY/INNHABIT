#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
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
    PeopleTracker tracker;

    // Use GPU for processing
    net.setPreferableBackend(DNN_BACKEND_CUDA);  // Set to CUDA backend
    net.setPreferableTarget(DNN_TARGET_CUDA);    // Set to use GPU (CUDA)

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
