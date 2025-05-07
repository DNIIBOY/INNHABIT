#include <tracker/utils.h>
#include <algorithm>

namespace tracker {

bool isInsideZone(const Position& person_position, const BoxZone& zone) {
    return (person_position.x >= zone.x1 && person_position.x <= zone.x2 &&
            person_position.y >= zone.y1 && person_position.y <= zone.y2);
}

bool intersectionOverArea(const BoxZone& bounding_box, const BoxZone& zone, const float threshold) {
    int x_left = std::max(bounding_box.x1, zone.x1);
    int y_top = std::max(bounding_box.y1, zone.y1);
    int x_right = std::min(bounding_box.x2, zone.x2);
    int y_btm = std::min(bounding_box.y2, zone.y2);

    //check if there is any overlap
    if (x_right < x_left || y_btm < y_top) {
        return false;
    }

    int intersection_area = (x_right - x_left) * (y_btm - y_top);
    int person_area = (bounding_box.x2 - bounding_box.x1) * (bounding_box.y2 - bounding_box.y1);
    if (person_area == 0) return false;

    return (static_cast<float>(intersection_area) / person_area) > threshold;
}

float computeIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int xA = std::max(box1.x, box2.x);
    int yA = std::max(box1.y, box2.y);
    int xB = std::min(box1.x + box1.width, box2.x + box2.width);
    int yB = std::min(box1.y + box1.height, box2.y + box2.height);

    int intersection_area = std::max(0, xB - xA) * std::max(0, yB - yA);
    int union_area = box1.area() + box2.area() - intersection_area;

    return union_area > 0 ? static_cast<float>(intersection_area) / union_area : 0.0f;
}

}  // namespace tracker