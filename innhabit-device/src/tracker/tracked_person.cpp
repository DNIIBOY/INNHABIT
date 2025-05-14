#include <tracker/tracked_person.h>
#include <tracker/utils.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>

namespace tracker {

TrackedPerson::TrackedPerson()
    : id_(0), class_id_(""), current_position_{0, 0}, postion_history_(), size_{0, 0},
      color_(0, 0, 0), missing_frames_(0), confidence_(0.0f), last_zone_type_(ZoneType::NONE),
      first_zone_type_(ZoneType::NONE) {}

TrackedPerson::TrackedPerson(int id, const Position& pos, const BoxSize& size, float conf,
                             int frame_height, const std::vector<TypedZone>& typed_zones)
    : id_(id), class_id_("person"), current_position_(pos), postion_history_({pos}), size_(size),
      color_(rand() % 255, rand() % 255, rand() % 255), missing_frames_(0), confidence_(conf),
      last_zone_type_(ZoneType::NONE), first_zone_type_(ZoneType::NONE) {
    initialize_zone_info(frame_height, typed_zones);
}

void TrackedPerson::initialize_zone_info(int frame_height, const std::vector<TypedZone>& typed_zones) {
    int top_y = current_position_.y - size_.height / 2;
    int bottom_y = current_position_.y + size_.height / 2;

    // Check if person appeared at top or bottom of frame
    bool from_top = top_y < frame_height * 0.1;
    bool from_bottom = bottom_y > frame_height * 0.9;

    // Check if person spawned in a zone
    BoxZone bounding_box = {current_position_.x - size_.width / 2,
                             current_position_.y - size_.height / 2,
                             current_position_.x + size_.width / 2,
                             current_position_.y + size_.height / 2};
    for (const auto& zone : typed_zones) {
        if (intersectionOverArea(bounding_box, zone.zone, 0.2f)) {
            first_zone_type_ = zone.type;
            last_zone_type_ = zone.type;
            break;
        }
    }
}

void TrackedPerson::update(const Position& pos, const BoxSize& size, float conf,
                           const std::vector<TypedZone>& typed_zones) {
    current_position_ = pos;
    size_ = size;
    confidence_ = conf;
    missing_frames_ = 0;

    // Update position history (limit to 30 entries)
    postion_history_.push_back(pos);
    if (postion_history_.size() > 30) {
        postion_history_.erase(postion_history_.begin());
    }

    // Update zone information
    bool in_any_zone = false;
    BoxZone bounding_box = {current_position_.x - size_.width / 2,
        current_position_.y - size_.height / 2,
        current_position_.x + size_.width / 2,
        current_position_.y + size_.height / 2};
    for (const auto& zone : typed_zones) {
        if (intersectionOverArea(bounding_box, zone.zone, 0.2f)) {
            last_zone_type_ = zone.type;
            in_any_zone = true;
            break;
        }
    }
    if (!in_any_zone && first_zone_type_ == ZoneType::NONE) {
        last_zone_type_ = ZoneType::NONE;
    }
}

cv::Rect TrackedPerson::getBoundingBox() const {
    return cv::Rect(current_position_.x - size_.width / 2,
                    current_position_.y - size_.height / 2,
                    size_.width, size_.height);
}

}  // namespace tracker
