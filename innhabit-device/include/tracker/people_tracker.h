#ifndef PEOPLE_TRACKER_H
#define PEOPLE_TRACKER_H

#include <memory>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>
#include "tracked_person.h"
#include "zone_type.h"
#include "configuration.h"

namespace tracker {

/**
 * @typedef MovementCallback
 * @brief Callback function type for movement events (e.g., entering/exiting zones).
 */
using MovementCallback = std::function<void(const TrackedPerson&, const std::string&)>;

/**
 * @class people_tracker
 * @brief Manages tracking of multiple persons in a video feed.
 */
class PeopleTracker {
public:
    /**
     * @brief Constructs a people_tracker with the given configuration.
     * @param config Shared pointer to the configuration object.
     */
    explicit PeopleTracker(std::shared_ptr<Configuration> config);

    /**
     * @brief Destructor for people_tracker.
     */
    ~();

    /**
     * @brief Updates the tracker with new detections.
     * @param detections List of detected objects.
     * @param frame_height Height of the video frame.
     */
    void update(const std::vector<Detection>& detections, int frame_height);

    /**
     * @brief Draws tracked persons and zones on the given frame.
     * @param frame The frame to draw on.
     */
    void draw(cv::Mat& frame);

    /**
     * @brief Gets the list of currently tracked persons.
     * @return Const reference to the vector of tracked persons.
     */
    const std::vector<TrackedPerson>& get_tracked_people() const;

    /**
     * @brief Sets the callback function for movement events.
     * @param callback The callback function to invoke on movement events.
     */
    void set_movement_callback(MovementCallback callback);

    /**
     * @brief Sets the access zones (entry/exit zones).
     * @param zones List of typed zones.
     */
    void set_access_zones(const std::vector<TypedZone>& zones);

private:
    std::shared_ptr<Configuration> config_; ///< Configuration object.
    std::vector<TrackedPerson> people_; ///< List of currently tracked persons.
    std::vector<TypedZone> access_zones_; ///< List of entry and exit zones.
    int next_id_; ///< Counter for assigning unique IDs to new persons.
    int max_missing_frames_; ///< Max frames a person can be missing before removal.
    float max_distance_; ///< Max distance for matching detections.
    float top_threshold_; ///< Threshold for detecting movement from top of frame.
    float bottom_threshold_; ///< Threshold for detecting movement from bottom of frame.
    MovementCallback movement_callback_; ///< Callback for movement events.

    /**
     * @brief Detects movement events (e.g., entering/exiting zones) and triggers callbacks.
     * @param person The tracked person to analyze.
     * @param frame_height Height of the video frame.
     * @param person_removed Whether the person was removed from tracking.
     */
    void detect_movements(TrackedPerson& person, int frame_height, bool person_removed = false);
};

}  // namespace tracker

#endif  // PEOPLE_TRACKER_H