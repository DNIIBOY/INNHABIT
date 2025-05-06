#ifndef TRACKER_UTILS_H
#define TRACKER_UTILS_H

#include <opencv2/opencv.hpp>
#include "zone_type.h"
#include "tracked_person.h"

namespace tracker {

/**
 * @brief Checks if a position is inside a given zone.
 * @param person_position The position to check.
 * @param zone The zone to check against.
 * @return True if the position is inside the zone, false otherwise.
 */
bool isInsideZone(const Position& person_position, const BoxZone& zone);

/**
 * @brief Computes the intersection over area ratio between two zones.
 * @param bounding_box The bounding box of the person.
 * @param zone The zone to check against.
 * @param threshold The threshold for considering the intersection significant.
 * @return True if the intersection over area exceeds the threshold, false otherwise.
 */
bool intersectionOverArea(const BoxZone& bounding_box, const BoxZone& zone, float threshold);

/**
 * @brief Computes the Intersection over Union (IoU) between two bounding boxes.
 * @param box1 The first bounding box.
 * @param box2 The second bounding box.
 * @return The IoU value (0.0 to 1.0).
 */
float computeIoU(const cv::Rect& box1, const cv::Rect& box2);

}  // namespace tracker

#endif  // TRACKER_UTILS_H