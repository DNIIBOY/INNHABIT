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

/**
 * @struct Position
 * @brief Structure representing the position of a tracking box.
 */
struct Position {
    int x; ///< X-coordinate of the position.
    int y; ///< Y-coordinate of the position.
};

/**
 * @struct BoxSize
 * @brief Structure representing the size of a tracking box.
 */
struct BoxSize {
    int width;  ///< Width of the box.
    int height; ///< Height of the box.
};

/**
 * @brief Checks if a given position is inside a specified bounding box.
 * @param pos The position to check.
 * @param zone The bounding box zone.
 * @return True if the position is inside the box, otherwise false.
 */
bool isInsideBox(const Position& pos, const BoxZone& zone);

/**
 * @class TrackedPerson
 * @brief Represents a tracked person in the system.
 */
class TrackedPerson {
public:
    int id; ///< Unique identifier for the tracked person.
    std::string classId; ///< Class ID from inference output.
    Position pos; ///< Current position of the tracked person.
    BoxSize size; ///< Size of the tracking box.
    std::vector<Position> history; ///< History of tracked positions.
    cv::Scalar color; ///< Color used for visualization.
    int missingFrames; ///< Number of frames the person has not been seen.
    float confidence; ///< Confidence score of the detection (0.0 - 1.0).
    bool fromTop; ///< Whether the person initially appeared in the top part of the image.
    bool fromBottom; ///< Whether the person initially appeared in the bottom part of the image.
    bool wasInZone; ///< Whether the person was last seen in an entrance zone.
    bool spawnedInZone; ///< Whether the person initially appeared in an entrance zone.

    /**
     * @brief Default constructor for TrackedPerson.
     */
    TrackedPerson();

    /**
     * @brief Parameterized constructor for TrackedPerson.
     * @param id The unique ID of the person.
     * @param position Initial position of the person.
     * @param boxSize Initial size of the tracking box.
     * @param conf Confidence score.
     * @param frameHeight Height of the frame.
     * @param entranceZones List of entrance zones.
     */
    TrackedPerson(int id, Position position, BoxSize boxSize, float conf, int frameHeight, 
                  const std::vector<BoxZone>& entranceZones);

    /**
     * @brief Updates the tracked person with new position, box size, and confidence.
     * @param position Updated position.
     * @param boxSize Updated box size.
     * @param conf Updated confidence score.
     */
    void update(Position position, BoxSize boxSize, float conf);

    /**
     * @brief Gets the bounding box from the current position and size.
     * @return The bounding box as a cv::Rect.
     */
    cv::Rect getBoundingBox() const;
};

/**
 * @typedef MovementCallback
 * @brief Callback function type for movement events.
 */
typedef void (*MovementCallback)(const TrackedPerson&, const std::string&);

/**
 * @class PeopleTracker
 * @brief Handles tracking of multiple persons.
 */
class PeopleTracker {
public:
    /**
     * @brief Constructor for PeopleTracker.
     * @param config Shared pointer to configuration.
     * @param maxMissingFrames Maximum allowed missing frames before removal.
     * @param maxDistance Maximum distance for matching detections.
     * @param topThreshold Threshold for detecting movement from the top.
     * @param bottomThreshold Threshold for detecting movement from the bottom.
     */
    PeopleTracker(std::shared_ptr<Configuration> config, int maxMissingFrames_ = 10, float maxDistance_ = 120.0f, 
                  float topThreshold_ = 0.1f, float bottomThreshold_ = 0.9f);

    /**
     * @brief Updates the tracker with new detections.
     * @param detections List of detected objects.
     * @param frameHeight Height of the frame.
     */
    void update(const std::vector<Detection>& detections, int frameHeight);

    /**
     * @brief Draws the tracked persons on the given frame.
     * @param frame The frame to draw on.
     */
    void draw(cv::Mat& frame);

    /**
     * @brief Gets the list of currently tracked persons.
     * @return A vector of tracked persons.
     */
    const std::vector<TrackedPerson>& getTrackedPeople() const;

    /**
     * @brief Sets the movement event callback function.
     * @param callback The callback function.
     */
    void setMovementCallback(MovementCallback callback);

    /**
     * @brief Sets the entrance zones for detecting movements.
     * @param zones A vector of entrance zones.
     */
    void setEntranceZones(const std::vector<BoxZone>& zones);

private:
    std::vector<TrackedPerson> people; ///< List of currently tracked people.
    int nextId; ///< ID counter for new tracked persons.
    int maxMissingFrames; ///< Maximum number of frames before a person is removed.
    float maxDistance; ///< Maximum allowable movement distance for matching.
    float topThreshold; ///< Threshold for movement from the top of the frame.
    float bottomThreshold; ///< Threshold for movement from the bottom of the frame.
    std::vector<BoxZone> entranceZones; ///< Entrance zones for tracking entry/exit events.
    MovementCallback movementCallback; ///< Callback function for movement events.

    /**
     * @brief Detects movement events and triggers callbacks.
     * @param person The tracked person to analyze.
     * @param frameHeight The height of the frame.
     * @param PersonRemoved Flag indicating if the person was removed.
     */
    void detectMovements(const TrackedPerson& person, int frameHeight, bool PersonRemoved = false);
};

#endif // TRACKER_H
