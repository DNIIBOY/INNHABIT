#include "tracker.h"
#include <algorithm>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

PeopleTracker::PeopleTracker(std::shared_ptr<Configuration> config, int maxMissingFrames_, 
    float maxDistance_, float topThreshold_, float bottomThreshold_)
: nextId(0), maxMissingFrames(maxMissingFrames_), maxDistance(maxDistance_),
topThreshold(topThreshold_), bottomThreshold(bottomThreshold_), // Fix typo if needed
movementCallback(nullptr) {
entranceZones = config->getEntranceZones();
}

void PeopleTracker::setEntranceZones(const std::vector<BoxZone>& zones) {
    entranceZones = zones;
}

bool isInsideBox(const Position& pos, const BoxZone& zone) {
    return (pos.x >= zone.x1 && pos.x <= zone.x2 && 
            pos.y >= zone.y1 && pos.y <= zone.y2);
}

float computeIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int xA = std::max(box1.x, box2.x);
    int yA = std::max(box1.y, box2.y);
    int xB = std::min(box1.x + box1.width, box2.x + box2.width);
    int yB = std::min(box1.y + box1.height, box2.y + box2.height);

    int intersectionArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    int unionArea = box1.area() + box2.area() - intersectionArea;

    return unionArea > 0 ? (float)intersectionArea / unionArea : 0.0f;
}


void PeopleTracker::update(const std::vector<Detection>& filteredDetections, int frameHeight) {
    std::vector<Detection> highConfDetections;
    std::vector<Detection> lowConfDetections;
    std::vector<bool> trackMatched(people.size(), false);
    std::vector<bool> detMatched(filteredDetections.size(), false);

    // ByteTrack parameters
    const float confThresholdHigh = 0.6f;
    const float confThresholdLow = 0.2f;
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
            float IoU = computeIoU(people[t].getBoundingBox(), highConfDetections[d].bounding_box);
            if (IoU > highestIoU && IoU > iouThresholdHigh) {
                highestIoU = IoU;
                bestDetIdx = d;
            }
        }

        if (bestDetIdx >= 0) {
            Position pos = {
                highConfDetections[bestDetIdx].bounding_box.x + highConfDetections[bestDetIdx].bounding_box.width / 2,
                highConfDetections[bestDetIdx].bounding_box.y + highConfDetections[bestDetIdx].bounding_box.height / 2
            };
            BoxSize size = {
                highConfDetections[bestDetIdx].bounding_box.width,
                highConfDetections[bestDetIdx].bounding_box.height
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
            float IoU = computeIoU(people[t].getBoundingBox(), lowConfDetections[d].bounding_box);
            if (IoU > highestIoU && IoU > iouThresholdLow) {
                highestIoU = IoU;
                bestDetIdx = d;
            }
        }

        if (bestDetIdx >= 0) {
            Position pos = {
                lowConfDetections[bestDetIdx].bounding_box.x + lowConfDetections[bestDetIdx].bounding_box.width / 2,
                lowConfDetections[bestDetIdx].bounding_box.y + lowConfDetections[bestDetIdx].bounding_box.height / 2
            };
            BoxSize size = {
                lowConfDetections[bestDetIdx].bounding_box.width,
                lowConfDetections[bestDetIdx].bounding_box.height
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
                detectMovements(people[t], frameHeight, true);  // Check movement before removal
            }
        } else {
            newPeople.push_back(people[t]);
        }
    }

    // Step 4: Add new tracks for unmatched high-confidence detections
    for (size_t d = 0; d < highConfDetections.size(); ++d) {
        if (!detMatched[d]) {
            Position pos = {
                highConfDetections[d].bounding_box.x + highConfDetections[d].bounding_box.width / 2,
                highConfDetections[d].bounding_box.y + highConfDetections[d].bounding_box.height / 2
            };
            BoxSize size = {
                highConfDetections[d].bounding_box.width,
                highConfDetections[d].bounding_box.height
            };
            TrackedPerson newPerson(nextId++, pos, size, highConfDetections[d].confidence, frameHeight, entranceZones);
            newPeople.push_back(newPerson);
        }
    }

    people = newPeople;
}

void PeopleTracker::detectMovements(const TrackedPerson& person, int frameHeight, bool PersonRemoved) {
    if (person.history.size() < 2 || !movementCallback) {
        return;
    }

    const Position& prevPos = person.history[person.history.size() - 2];
    const Position& currPos = person.history.back();

    // Create a mutable copy of the person to update wasInZone
    TrackedPerson& mutablePerson = const_cast<TrackedPerson&>(person);

    for (size_t i = 0; i < entranceZones.size(); ++i) {
        bool prevInZone = isInsideBox(prevPos, entranceZones[i]);
        bool currInZone = isInsideBox(currPos, entranceZones[i]);

        // Update wasInZone based on current position
        if (currInZone) {
            mutablePerson.wasInZone = true;
        }

        // Entry event: Person spawned in zone and moved out
        if (person.spawnedInZone && !currInZone && PersonRemoved) {
            movementCallback(mutablePerson, "entered");
            mutablePerson.wasInZone = false; // Reset after event
            return;
        }

        // Exit event: Person is removed while in zone
        if (PersonRemoved && mutablePerson.wasInZone && prevInZone) {
            movementCallback(mutablePerson, "exited");
            mutablePerson.wasInZone = false;
            return;
        }
    }

    // If not in any zone currently and wasn't spawned in a zone, reset wasInZone
    bool inAnyZone = false;
    for (const auto& zone : entranceZones) {
        if (isInsideBox(currPos, zone)) {
            inAnyZone = true;
            break;
        }
    }
    if (!inAnyZone && !person.spawnedInZone) {
        mutablePerson.wasInZone = false;
    }
}

void PeopleTracker::draw(cv::Mat& frame) {
    for (const auto& person : people) {
        cv::Rect bbox = person.getBoundingBox();
        cv::rectangle(frame, bbox, person.color, 2);

        // Include confidence in the label
        string label = "ID: " + to_string(person.id) + " Conf: " + to_string(person.confidence);
        if (person.wasInZone) {
            label += " (In Zone)";
        }
        if (person.spawnedInZone) {
            label += " (Spawned in Zone)";
        }
        int y = max(bbox.y - 10, 15);
        cv::putText(frame, label, cv::Point(bbox.x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, person.color, 2);

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

    for (size_t i = 0; i < entranceZones.size(); ++i) {
        const auto& zone = entranceZones[i];
        cv::rectangle(frame, cv::Point(zone.x1, zone.y1), cv::Point(zone.x2, zone.y2),
                     cv::Scalar(0, 255 - (i * 50 % 255), i * 50 % 255), 2);
        cv::putText(frame, "Zone " + to_string(i), cv::Point(zone.x1, zone.y1 - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255 - (i * 50 % 255), i * 50 % 255), 2);
    }
}
const std::vector<TrackedPerson>& PeopleTracker::getTrackedPeople() const {
    return people;
}

void PeopleTracker::setMovementCallback(MovementCallback callback) {
    movementCallback = callback;
}
