#ifndef TRACKED_PERSON_H
#define TRACKED_PERSON_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "zone_type.h"

namespace tracker {

/**
 * @struct Position
 * @brief Represents the 2D position of a tracked person.
 */
struct Position {
    int x; ///< X-coordinate.
    int y; ///< Y-coordinate.
};

/**
 * @struct BoxSize
 * @brief Represents the size of a bounding box.
 */
struct BoxSize {
    int width;  ///< Width of the box.
    int height; ///< Height of the box.
};

/**
 * @class TrackedPerson
 * @brief Represents a single tracked person in the system.
 */
class TrackedPerson {
public:
    /**
     * @brief Default constructor for TrackedPerson.
     */
    TrackedPerson();

    /**
     * @brief Constructs a TrackedPerson with initial parameters.
     * @param id Unique identifier for the person.
     * @param pos Initial position.
     * @param size Initial bounding box size.
     * @param conf Confidence score of the detection.
     * @param frame_height Height of the video frame.
     * @param typed_zones List of entry/exit zones.
     */
    TrackedPerson(int id, const Position& pos, const BoxSize& size, float conf,
                  int frame_height, const std::vector<TypedZone>& typed_zones);

    /**
     * @brief Updates the person's state with new detection data.
     * @param pos Updated position.
     * @param size Updated bounding box size.
     * @param conf Updated confidence score.
     * @param typed_zones List of entry/exit zones.
     */
    void update(const Position& pos, const BoxSize& size, float conf,
                const std::vector<TypedZone>& typed_zones);

    /**
     * @brief Gets the bounding box for the person.
     * @return The bounding box as a cv::Rect.
     */
    cv::Rect getBoundingBox() const;

    int id_; ///< Unique identifier.
    std::string class_id_; ///< Class ID (e.g., "person").
    Position current_position_; ///< Current position.
    std::vector<Position> postion_history_; ///< History of positions.
    BoxSize size_; ///< Bounding box size.
    cv::Scalar color_; ///< Color for visualization.
    int missing_frames_; ///< Frames since last detection.
    float confidence_; ///< Detection confidence score.
    ZoneType last_zone_type_; ///< Type of the last zone the person was in.
    ZoneType first_zone_type_; ///< Type of the zone where the person first appeared.

private:
    /**
     * @brief Determines if the person appeared at the top or bottom of the frame /

System: and updates zone information.
     * @param frame_height Height of the video frame.
     * @param typed_zones List of entry/exit zones.
     */
    void initialize_zone_info(int frame_height, const std::vector<TypedZone>& typed_zones);
};

}  // namespace tracker

#endif  // TRACKED_PERSON_H