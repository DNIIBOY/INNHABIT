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
struct TrackerInfo {
    Ptr<Tracker> tracker;
    Rect box;
    Scalar color;
    int age;  // Age of the tracker (in frames)
};

int main() {
    // Load YOLO model and initialize webcam
    Net net = readNet("yolov7-tiny.weights", "yolov7-tiny.cfg");

    // Load class labels (COCO dataset labels)
    vector<string> classes;
    ifstream ifs("coco.names");
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }

    // Open the webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open webcam" << endl;
        return -1;
    }

    // Get YOLO output layer names
    vector<string> layerNames = net.getUnconnectedOutLayersNames();

    // This will store the tracker information
    vector<TrackerInfo> trackers;

    // This will store the detected bounding boxes
    vector<Rect> detectedBoxes;

    // Tracking parameters
    const int maxTrackerAge = 30; // Number of frames to keep a lost tracker before removal

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
        vector<Rect> detectedBoxesNew;

        for (auto& output : outs) {
            for (int i = 0; i < output.rows; i++) {
                float* data = output.ptr<float>(i);
                vector<float> scores(data + 5, data + output.cols);
                int classId = max_element(scores.begin(), scores.end()) - scores.begin();
                float confidence = scores[classId];
                if (confidence > 0.5 && classes[classId] == "person") {  // Check if it's a person
                    int centerX = static_cast<int>(data[0] * width);
                    int centerY = static_cast<int>(data[1] * height);
                    int w = static_cast<int>(data[2] * width);
                    int h = static_cast<int>(data[3] * height);
                    detectedBoxesNew.emplace_back(centerX - w / 2, centerY - h / 2, w, h);
                    confidences.push_back(confidence);
                    classIds.push_back(classId);
                }
            }
        }

        // Non-maxima suppression
        vector<int> indices;
        NMSBoxes(detectedBoxesNew, confidences, 0.5, 0.4, indices);

        // Update detected bounding boxes
        detectedBoxes.clear();
        for (int i : indices) {
            detectedBoxes.push_back(detectedBoxesNew[i]);
        }

        // Update trackers and verify with YOLO detection
        vector<bool> trackerMatched(detectedBoxes.size(), false);
        for (auto& trackerInfo : trackers) {
            // Try to update existing trackers
            bool ok = trackerInfo.tracker->update(frame, trackerInfo.box);
            if (ok) {
                // Draw tracker box with its persistent color
                rectangle(frame, trackerInfo.box, trackerInfo.color, 2, 1);
                trackerInfo.age = 0; // Reset the age if the tracker is still valid
                
                // Verify if the tracker is still tracking a person using YOLO
                Mat trackerROI = frame(trackerInfo.box);  // Extract the region of interest (ROI) of the tracker
                Mat trackerBlob;
                blobFromImage(trackerROI, trackerBlob, 0.00392, Size(320, 320), Scalar(0, 0, 0), true, false);
                net.setInput(trackerBlob);
                vector<Mat> outsROI;
                net.forward(outsROI, layerNames);

                // Post-process the tracker ROI using YOLO to check if it’s still a person
                bool isPersonTracked = false;
                for (auto& output : outsROI) {
                    for (int i = 0; i < output.rows; i++) {
                        float* data = output.ptr<float>(i);
                        vector<float> scores(data + 5, data + output.cols);
                        int classId = max_element(scores.begin(), scores.end()) - scores.begin();
                        float confidence = scores[classId];
                        if (confidence > 0.5 && classes[classId] == "person") {
                            isPersonTracked = true;
                            break;
                        }
                    }
                }

                // If the tracker is not tracking a person, reset the tracker
                if (!isPersonTracked) {
                    trackerInfo.tracker = TrackerKCF::create();  // Reset the tracker
                    trackerInfo.age = maxTrackerAge + 1;  // Force tracker removal
                }
            } else {
                // Increment the age if tracking failed
                trackerInfo.age++;
            }
        }

        // Remove trackers that have been lost for too long
        trackers.erase(remove_if(trackers.begin(), trackers.end(),
                                  [](const TrackerInfo& trackerInfo) { return trackerInfo.age > maxTrackerAge; }),
                       trackers.end());

        // Create new trackers for new detections
        for (size_t i = 0; i < detectedBoxes.size(); ++i) {
            bool matched = false;
            for (size_t j = 0; j < trackers.size(); ++j) {
                // If the new detection overlaps an existing tracker, update the tracker
                if (!trackerMatched[i] && (detectedBoxes[i] & trackers[j].box).area() > 0.3 * detectedBoxes[i].area()) {
                    trackerMatched[i] = true;
                    break;
                }
            }

            if (!trackerMatched[i]) {
                // Create a new tracker for the new detection
                Ptr<TrackerKCF> newTracker = TrackerKCF::create();
                newTracker->init(frame, detectedBoxes[i]);

                // Assign a random color to the new tracker
                Scalar color(rand() % 256, rand() % 256, rand() % 256);

                // Add to the trackers list
                trackers.push_back(TrackerInfo{newTracker, detectedBoxes[i], color, 0});
            }
        }

        // Display the output frame
        imshow("Tracking", frame);

        // Print the number of detected people
        cout << "Number of detected people: " << detectedBoxes.size() << endl;

        // Wait for 'q' key to break the loop
        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
