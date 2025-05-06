#include "tracker.h"
#include <algorithm>
#include <iostream>
#include <cmath>

bool isInsideBox(const Position& pos, const BoxZone& zone) {
    return (pos.x >= zone.x1 && pos.x <= zone.x2 && 
            pos.y >= zone.y1 && pos.y <= zone.y2);
}

bool intersectionOverArea(const BoxZone& personZone, const BoxZone& entryZone, const float isInsideBoxThreshold) {
    int x_left = std::max(personZone.x1, entryZone.x1);
    int y_top = std::max(personZone.y1, entryZone.y1);
    int x_right = std::min(personZone.x2, entryZone.x2);
    int y_btm = std::min(personZone.y2, entryZone.y2);

    if (x_right < x_left || y_btm < y_top) {
        return false;
    }

    int intersectionArea = (x_right - x_left) * (y_btm - y_top);
    int personArea = (personZone.x2 - personZone.x1) * (personZone.y2 - personZone.y1);
    if (personArea == 0) return false;

    return (static_cast<float>(intersectionArea) / personArea) > isInsideBoxThreshold;
}

float computeIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int xA = std::max(box1.x, box2.x);
    int yA = std::max(box1.y, box2.y);
    int xB = std::min(box1.x + box1.width, box2.x + box2.width);
    int yB = std::min(box1.y + box1.height, box2.y + box2.height);

    int intersectionArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    int unionArea = box1.area() + box2.area() - intersectionArea;

    return unionArea > 0 ? static_cast<float>(intersectionArea) / unionArea : 0.0f;
}

TrackedPerson::TrackedPerson() : id(0), classId(""), missingFrames(0), confidence(0.0f), 
                                fromTop(false), fromBottom(false), wasInZone(false), 
                                spawnedInZone(false) {}

TrackedPerson::TrackedPerson(int id, Position position, BoxSize boxSize, float conf, 
                             int frameHeight, const std::vector<BoxZone>& entranceZones)
    : id(id), classId("person"), missingFrames(0), confidence(conf), 
      fromTop(false), fromBottom(false), wasInZone(false), spawnedInZone(false) {
    update(position, boxSize, conf, entranceZones);
    color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
    
    int topY = position.y - boxSize.height / 2;
    int bottomY = position.y + boxSize.height / 2;
    fromTop = topY < frameHeight * 0.1;
    fromBottom = bottomY > frameHeight * 0.9;

    for (const auto& zone : entranceZones) {
        if (isInsideBox(position, zone)) {
            spawnedInZone = true;
            wasInZone = true;
            break;
        }
    }
}

void TrackedPerson::update(Position position, BoxSize boxSize, float conf, 
                           const std::vector<BoxZone>& entranceZones) {
    pos = position;
    size = boxSize;
    history.push_back(position);
    if (history.size() > 30) {
        history.erase(history.begin());
    }
    missingFrames = 0;
    confidence = conf;
    bool inAnyZone = false;
    for (const auto& zone : entranceZones) {
        if (isInsideBox(position, zone)) {
            inAnyZone = true;
            wasInZone = true;
            break;
        }
    }
    if (!inAnyZone && !spawnedInZone) {
        wasInZone = false;
    }
}

cv::Rect TrackedPerson::getBoundingBox() const {
    return cv::Rect(pos.x - size.width / 2, pos.y - size.height / 2, size.width, size.height);
}

PeopleTracker::PeopleTracker(std::shared_ptr<Configuration> config, int maxMissingFrames, 
                            float maxDistance, float topThreshold, float bottomThreshold)
    : nextId(0), maxMissingFrames(maxMissingFrames), maxDistance(maxDistance),
      topThreshold(topThreshold), bottomThreshold(bottomThreshold), movementCallback(nullptr) {
    m_config = config;
    entranceZones = config->getEntranceZones();

    // Log entrance zones for debugging
    LOG("Entrance zones: " << (entranceZones.empty() ? "None" : ""));
    for (size_t i = 0; i < entranceZones.size(); ++i) {
        LOG("Zone " << i << ": (x1=" << entranceZones[i].x1 << ", y1=" << entranceZones[i].y1 
            << ", x2=" << entranceZones[i].x2 << ", y2=" << entranceZones[i].y2 << ")");
    }
    if (entranceZones.empty()) {
        ERROR("No entrance zones configured");
    }
}

void PeopleTracker::setEntranceZones(const std::vector<BoxZone>& zones) {
    entranceZones = zones;
}

void PeopleTracker::update(const std::vector<Detection>& filteredDetections, int frameHeight) {
    if (entranceZones.empty()) {
        ERROR("Cannot update tracker: no entrance zones configured");
        return;
    }

    std::vector<Detection> highConfDetections;
    std::vector<Detection> lowConfDetections;
    std::vector<bool> trackMatched(people.size(), false);
    std::vector<bool> detMatched(filteredDetections.size(), false);

    const float confThresholdHigh = 0.6f;
    const float confThresholdLow = 0.2f;
    const float iouThresholdHigh = 0.3f;
    const float iouThresholdLow = 0.4f;

    for (size_t i = 0; i < filteredDetections.size(); ++i) {
        if (filteredDetections[i].confidence > confThresholdHigh) {
            highConfDetections.push_back(filteredDetections[i]);
        } else if (filteredDetections[i].confidence > confThresholdLow) {
            lowConfDetections.push_back(filteredDetections[i]);
        }
    }

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
            people[t].update(pos, size, highConfDetections[bestDetIdx].confidence, entranceZones);
            trackMatched[t] = true;
            detMatched[bestDetIdx] = true;
            detectMovements(people[t], frameHeight);
        }
    }

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
            people[t].update(pos, size, lowConfDetections[bestDetIdx].confidence, entranceZones);
            trackMatched[t] = true;
            detMatched[bestDetIdx + highConfDetections.size()] = true;
            detectMovements(people[t], frameHeight);
        }
    }

    std::vector<TrackedPerson> newPeople;
    for (size_t t = 0; t < people.size(); ++t) {
        if (!trackMatched[t]) {
            people[t].missingFrames++;
            if (people[t].missingFrames < maxMissingFrames) {
                newPeople.push_back(people[t]);
            } else {
                detectMovements(people[t], frameHeight, true);
            }
        } else {
            newPeople.push_back(people[t]);
        }
    }

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
            TrackedPerson newPerson(nextId++, pos, size, highConfDetections[d].confidence, 
                                   frameHeight, entranceZones);
            newPeople.push_back(newPerson);
            detectMovements(newPerson, frameHeight);
        }
    }

    people = newPeople;
}

void PeopleTracker::detectMovements(TrackedPerson& person, int frameHeight, bool PersonRemoved) {
    if (person.history.size() < 2 || !movementCallback) {
        return;
    }

    const Position& prevPos = person.history[person.history.size() - 2];
    const Position& currPos = person.history.back();

    for (size_t i = 0; i < entranceZones.size(); ++i) {
        bool prevInZone = isInsideBox(prevPos, entranceZones[i]);
        bool currInZone = isInsideBox(currPos, entranceZones[i]);

        if (currInZone) {
            person.wasInZone = true;
        }

        if (person.spawnedInZone && !currInZone && PersonRemoved) {
            movementCallback(person, "entered");
            person.wasInZone = false;
            return;
        }

        if (PersonRemoved && person.wasInZone && prevInZone) {
            movementCallback(person, "exited");
            person.wasInZone = false;
            return;
        }
    }

    bool inAnyZone = false;
    for (const auto& zone : entranceZones) {
        if (isInsideBox(currPos, zone)) {
            inAnyZone = true;
            break;
        }
    }
    if (!inAnyZone && !person.spawnedInZone) {
        person.wasInZone = false;
    }
}

void PeopleTracker::draw(cv::Mat& frame) {
    if (entranceZones.empty()) {
        ERROR("Cannot draw: no entrance zones configured");
        return;
    }

    for (size_t i = 0; i < entranceZones.size(); ++i) {
        const auto& zone = entranceZones[i];
        cv::rectangle(frame, cv::Point(zone.x1, zone.y1), cv::Point(zone.x2, zone.y2),
                      cv::Scalar(0, 255 - (i * 50 % 255), i * 50 % 255), 2);
        cv::putText(frame, "Zone " + std::to_string(i), cv::Point(zone.x1, zone.y1 - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255 - (i * 50 % 255), i * 50 % 255), 2);
    }

    for (const auto& person : people) {
        cv::Rect bbox = person.getBoundingBox();
        cv::rectangle(frame, bbox, person.color, 2);

        std::string label = "ID: " + std::to_string(person.id) + " Conf: " + std::to_string(person.confidence);
        if (person.wasInZone) {
            label += " (In Zone)";
        }
        if (person.spawnedInZone) {
            label += " (Spawned in Zone)";
        }
        int y = std::max(bbox.y - 10, 15);
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
}

const std::vector<TrackedPerson>& PeopleTracker::getTrackedPeople() const {
    return people;
}

void PeopleTracker::setMovementCallback(MovementCallback callback) {
    movementCallback = callback;
}