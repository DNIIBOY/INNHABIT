#ifndef TRACKER_H
#define TRACKER_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cmath>
#include "detector.h"
#include "common.h"

using namespace cv;
using namespace std;

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

// Forward declaration of isInsideBox function
bool isInsideBox(const Position& pos, const BoxZone& zone);

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
    bool wasInZone;      // True if last seen in any entrance zone
    bool spawnedInZone;  // True if the person initially spawned in an entrance zone
    
    TrackedPerson() : id(-1), missingFrames(0), confidence(0.0f), fromTop(false), fromBottom(false), 
                      wasInZone(false), spawnedInZone(false) {}
    
    TrackedPerson(int id, Position position, BoxSize boxSize, float conf, int frameHeight, 
                  const std::vector<BoxZone>& entranceZones) {
        this->id = id;
        this->classId = "person";
        this->update(position, boxSize, conf);
        this->color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        
        int topY = position.y - boxSize.height / 2;
        int bottomY = position.y + boxSize.height / 2;
        this->fromTop = topY < frameHeight * 0.1;
        this->fromBottom = bottomY > frameHeight * 0.9;
        this->wasInZone = false;
        this->spawnedInZone = false;
        // Check if the initial position is inside any entrance zone
        for (const auto& zone : entranceZones) {
            if (isInsideBox(position, zone)) {
                this->spawnedInZone = true;
                this->wasInZone = true;
                break;
            }
        }
    }
    
    void update(Position position, BoxSize boxSize, float conf) {
        this->pos = position;
        this->size = boxSize;
        this->history.push_back(position);
        if (this->history.size() > 30) {
            this->history.erase(this->history.begin());
        }
        this->missingFrames = 0;
        this->confidence = conf;
        // Note: wasInZone is not updated here; it’s updated in detectMovements
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
    PeopleTracker(int maxMissingFrames = 10, float maxDistance = 120.0f, 
                 float topThreshold = 0.1f, float bottomThreshold = 0.9f);
    
    void update(const std::vector<Detection>& detections, int frameHeight);
    void draw(cv::Mat& frame);
    const std::vector<TrackedPerson>& getTrackedPeople() const;
    void setMovementCallback(MovementCallback callback);
    void setEntranceZones(const std::vector<BoxZone>& zones);

private:
    std::vector<TrackedPerson> people;
    int nextId;
    int maxMissingFrames;
    float maxDistance;
    float topThreshold;
    float bottomThreshold;
    std::vector<BoxZone> entranceZones;
    MovementCallback movementCallback;
    
    void detectMovements(const TrackedPerson& person, int frameHeight, bool PersonRemoved = false);
};

#endif // TRACKER_H