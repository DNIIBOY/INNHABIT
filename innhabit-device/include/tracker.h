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
/**
    * Structure used for storing position of tracking box
*/
struct Position {
    int x;
    int y;
};

/**
    * Structure used for storing size of tracking box
*/
struct BoxSize {
    int width;
    int height;
};

// Forward declaration of isInsideBox function
bool isInsideBox(const Position& pos, const BoxZone& zone);

/**
    * Class representing a tracked person
*/
class TrackedPerson {
public:
    /** Id used to distingues multiple TrackedPerson */
    int id;
    /** Id from inference output classifying type of object */
    std::string classId;
    Position pos;
    BoxSize size;
    /** vector<position> used to store previous and current positions of TrackedPerson */
    std::vector<Position> history;
    cv::Scalar color;
    /** Amount of frames TrackedPerson has not been seen */
    int missingFrames;
    /** float ranging 0.0-1.0 describing the confidence of the detection */
    float confidence;
    /** True if TrackedPerson initially spawned in top part of image */
    bool fromTop;
    /** True if TrackedPerson initially spawned in top part of image */
    bool fromBottom;
    /** True if TrackedPerson last seen in any entrance zone */
    bool wasInZone;
    /** True if TrackedPerson initially spawned in an entrance zone */
    bool spawnedInZone; 
    
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
    
    /** 
    * Updates the TrackedPerson with new postion boxSize and conf
    * wasInZone is not updated here, but in detectMovements
    */
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
    
    /**
    * Gets cv::Rect from Position and BoxSize
    */
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

/** 
    * PeopleTracker class, handles multiple instances of TrackedPerson
    */
class PeopleTracker {
public:
    /**
    * Contructor for PeopleTracker
    */
    PeopleTracker(std::shared_ptr<Configuration> config, int maxMissingFrames_ = 10, float maxDistance_ = 120.0f, 
                 float topThreshold_ = 0.1f, float bottomThreshold_ = 0.9f);
    /**
    * Updates loops over all detection from latest frame.
    * Step 1: Match high-confidence detections to existing tracks
    * Step 2: Match unmatched tracks to low-confidence detections
    * Step 3: Update missing frames and remove old tracks
    * Step 4: Add new tracks for unmatched high-confidence detections
    */
    void update(const std::vector<Detection>& detections, int frameHeight);
    /**Draws all TrackedPerson on a given fram*/
    void draw(cv::Mat& frame);
    /** Vector containing all currently TrackedPerson*/
    const std::vector<TrackedPerson>& getTrackedPeople() const;
    void setMovementCallback(MovementCallback callback);
    /** Sets the entranceZones for the given frame, Used to determine if TrackedPerson has exited or entered*/
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
