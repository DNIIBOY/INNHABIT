#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cmath>
#include "detector.h"

// Position structure for tracking
struct Position {
    int x;
    int y;
};

// Box size structure
struct BoxSize {
    int width;
    int height;
};

// Structure representing a tracked person
class TrackedPerson {
public:
    int id;
    std::string classId;
    Position pos;
    BoxSize size;
    std::vector<Position> history;
    cv::Scalar color;
    int missingFrames;
    float confidence;
    bool fromTop;
    bool fromBottom;
    
    TrackedPerson() : id(-1), missingFrames(0), confidence(0.0f), fromTop(false), fromBottom(false) {}
    
    TrackedPerson(int id, Position position, BoxSize boxSize, float conf, int frameHeight) {
        this->id = id;
        this->classId = "person";
        this->update(position, boxSize, conf);
        this->color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        
        // Determine entry direction
        int topY = position.y - boxSize.height / 2;
        int bottomY = position.y + boxSize.height / 2;
        this->fromTop = topY < frameHeight * 0.1;  // Top 10% of frame
        this->fromBottom = bottomY > frameHeight * 0.9;  // Bottom 10% of frame
    }
    
    void update(Position position, BoxSize boxSize, float conf) {
        this->pos = position;
        this->size = boxSize;
        this->history.push_back(position);
        if (this->history.size() > 30) {  // Limit history length
            this->history.erase(this->history.begin());
        }
        this->missingFrames = 0;
        this->confidence = conf;
    }
    
    cv::Rect getBoundingBox() const {
        return cv::Rect(
            pos.x - size.width / 2,
            pos.y - size.height / 2,
            size.width,
            size.height
        );
    }
};

// Movement event callback function type
typedef void (*MovementCallback)(const TrackedPerson&, const std::string&);

// People tracker class
class PeopleTracker {
public:
    PeopleTracker(int maxMissingFrames = 10, float maxDistance = 120.0f, float topThreshold = 0.1f, float bottomThreshold = 0.9f);
    
    // Update tracker with new detections
    void update(const std::vector<Detection>& detections, int frameHeight);
    
    // Draw tracking information on frame
    void draw(cv::Mat& frame);
    
    // Get current tracked people
    const std::vector<TrackedPerson>& getTrackedPeople() const;
    
    // Set callback for movement events
    void setMovementCallback(MovementCallback callback);
    
private:
    std::vector<TrackedPerson> people;
    int nextId;
    int maxMissingFrames;
    float maxDistance;
    float topThreshold;
    float bottomThreshold;
    MovementCallback movementCallback;
    
    // Detect person movements between zones
    void detectMovements(const TrackedPerson& person, int frameHeight);
};

#endif // TRACKER_H