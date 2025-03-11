#include "tracker.h"
#include <algorithm>
#include <iostream>
#include <cmath>

using namespace cv;  // Add OpenCV namespace
using namespace std; // Add standard namespace

PeopleTracker::PeopleTracker(int maxMissingFrames_, float maxDistance_, float topThreshold_, float bottomThreshold_)
    : nextId(0), maxMissingFrames(maxMissingFrames_), maxDistance(maxDistance_),
      topThreshold(topThreshold_), bottomThreshold(bottomThreshold_) /* Fixed typo */, movementCallback(nullptr) {
}

float computeIoU(const cv::Rect& box1, const cv::Rect& box2) {
    float intersection = (box1 & box2).area();
    float unionArea = box1.area() + box2.area() - intersection;
    return unionArea > 0 ? intersection / unionArea : 0;
}

void PeopleTracker::update(const std::vector<Detection>& detections, int frameHeight) {
    std::vector<Detection> filteredDetections;
    for (const auto& det : detections) {
        if (det.classId == "person") filteredDetections.push_back(det);  // Only track people
    }

    std::vector<Detection> highConfDetections;
    std::vector<Detection> lowConfDetections;
    std::vector<bool> trackMatched(people.size(), false);
    std::vector<bool> detMatched(filteredDetections.size(), false);

    // ByteTrack parameters
    const float confThresholdHigh = 0.6f;
    const float confThresholdLow = 0.1f;
    const float iouThresholdHigh = 0.3f;
    const float iouThresholdLow = 0.4f;

    // Separate high and low-confidence detections
    for (size_t i = 0; i < filteredDetections.size(); ++i) {
        if (filteredDetections[i].confidence > confThresholdHigh) {
            highConfDetections.push_back(filteredDetections[i]);
        } else if (filteredDetections[i].confidence > confThresholdLow) {
            lowConfDetections.push_back(filteredDetections[i]);
        }
    }

    // Step 1: Match high-confidence detections to existing tracks
    for (size_t t = 0; t < people.size(); ++t) {
        float highestIoU = 0.0;
        int bestDetIdx = -1;

        for (size_t d = 0; d < highConfDetections.size(); ++d) {
            if (detMatched[d]) continue;
            float IoU = computeIoU(people[t].getBoundingBox(), highConfDetections[d].box);
            if (IoU > highestIoU && IoU > iouThresholdHigh) {
                highestIoU = IoU;
                bestDetIdx = d;
            }
        }

        if (bestDetIdx >= 0) {
            Position pos = {
                highConfDetections[bestDetIdx].box.x + highConfDetections[bestDetIdx].box.width / 2,
                highConfDetections[bestDetIdx].box.y + highConfDetections[bestDetIdx].box.height / 2
            };
            BoxSize size = {
                highConfDetections[bestDetIdx].box.width,
                highConfDetections[bestDetIdx].box.height
            };
            people[t].update(pos, size, highConfDetections[bestDetIdx].confidence);
            trackMatched[t] = true;
            detMatched[bestDetIdx] = true;
            detectMovements(people[t], frameHeight);
        }
    }

    // Step 2: Match unmatched tracks to low-confidence detections
    for (size_t t = 0; t < people.size(); ++t) {
        if (trackMatched[t]) continue;

        float highestIoU = 0.0;
        int bestDetIdx = -1;

        for (size_t d = 0; d < lowConfDetections.size(); ++d) {
            if (detMatched[d + highConfDetections.size()]) continue;
            float IoU = computeIoU(people[t].getBoundingBox(), lowConfDetections[d].box);
            if (IoU > highestIoU && IoU > iouThresholdLow) {
                highestIoU = IoU;
                bestDetIdx = d;
            }
        }

        if (bestDetIdx >= 0) {
            Position pos = {
                lowConfDetections[bestDetIdx].box.x + lowConfDetections[bestDetIdx].box.width / 2,
                lowConfDetections[bestDetIdx].box.y + lowConfDetections[bestDetIdx].box.height / 2
            };
            BoxSize size = {
                lowConfDetections[bestDetIdx].box.width,
                lowConfDetections[bestDetIdx].box.height
            };
            people[t].update(pos, size, lowConfDetections[bestDetIdx].confidence);
            trackMatched[t] = true;
            detMatched[bestDetIdx + highConfDetections.size()] = true;
            detectMovements(people[t], frameHeight);
        }
    }

    // Step 3: Update missing frames and remove old tracks
    std::vector<TrackedPerson> newPeople;
    for (size_t t = 0; t < people.size(); ++t) {
        if (!trackMatched[t]) {
            people[t].missingFrames++;
            if (people[t].missingFrames < maxMissingFrames) {
                newPeople.push_back(people[t]);
            } else {
                detectMovements(people[t], frameHeight);  // Check movement before removal

            }
        } else {
            newPeople.push_back(people[t]);
        }
    }

    // Step 4: Add new tracks for unmatched high-confidence detections
    for (size_t d = 0; d < highConfDetections.size(); ++d) {
        if (!detMatched[d]) {
            Position pos = {
                highConfDetections[d].box.x + highConfDetections[d].box.width / 2,
                highConfDetections[d].box.y + highConfDetections[d].box.height / 2
            };
            BoxSize size = {
                highConfDetections[d].box.width,
                highConfDetections[d].box.height
            };
            TrackedPerson newPerson(nextId++, pos, size, highConfDetections[d].confidence, frameHeight);
            newPeople.push_back(newPerson);
        }
    }

    people = newPeople;
}

void PeopleTracker::detectMovements(const TrackedPerson& person, int frameHeight) {
    if (person.history.size() < 5 || !movementCallback) return;

    const Position& start = person.history.front();
    const Position& end = person.history.back();

    int startY = start.y;
    int endY = end.y;

    if (person.fromTop && endY > frameHeight * bottomThreshold) {
        movementCallback(person, "exit");
    } else if (person.fromBottom && endY < frameHeight * topThreshold) {
        movementCallback(person, "enter");
    }
}

void PeopleTracker::draw(cv::Mat& frame) {
    for (const auto& person : people) {
        cv::Rect bbox = person.getBoundingBox();
        cv::rectangle(frame, bbox, person.color, 2);

        string label = "ID: " + to_string(person.id);
        if (person.fromTop) label += " (Top)";
        else if (person.fromBottom) label += " (Bottom)";
        int y = max(bbox.y - 10, 15);
        cv::putText(frame, label, cv::Point(bbox.x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, person.color, 2);

        // Draw movement trail
        if (person.history.size() > 1) {
            for (size_t i = 1; i < person.history.size(); ++i) {
                float alpha = static_cast<float>(i) / person.history.size();
                cv::Scalar trailColor = person.color * alpha;
                cv::line(frame, cv::Point(person.history[i-1].x, person.history[i-1].y),
                         cv::Point(person.history[i].x, person.history[i].y), trailColor, 1);
                cv::circle(frame, cv::Point(person.history[i].x, person.history[i].y), 2, trailColor, -1);
            }
        }
    }

    // Draw entry/exit zones (optional)
    int topZoneY = frame.rows * topThreshold;
    int bottomZoneY = frame.rows * bottomThreshold;
}

const std::vector<TrackedPerson>& PeopleTracker::getTrackedPeople() const {
    return people;
}

void PeopleTracker::setMovementCallback(MovementCallback callback) {
    movementCallback = callback;
}