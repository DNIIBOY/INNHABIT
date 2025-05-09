#include <tracker/people_tracker.h>
#include <tracker/utils.h>
#include <opencv2/opencv.hpp>
#include "detector.h"
#include <algorithm>
#include <iostream>

namespace tracker {

PeopleTracker::PeopleTracker(std::shared_ptr<Configuration> config)
    : config_(config), people_(), access_zones_(config->GetTypedZones()), next_id_(0),
      max_missing_frames_(10), max_distance_(120.0f), top_threshold_(0.1f), bottom_threshold_(0.9f),
      movement_callback_(nullptr) {
    // Log access zones for debugging
    LOG("Access zones: " << (access_zones_.empty() ? "None" : ""));
    for (size_t i = 0; i < access_zones_.size(); ++i) {
        const auto& zone = access_zones_[i];
        LOG("Zone " << i << ": (x1=" << zone.zone.x1 << ", y1=" << zone.zone.y1
            << ", x2=" << zone.zone.x2 << ", y2=" << zone.zone.y2 << ", type=" << zone.getTypeString() << ")");
    }
    if (access_zones_.empty()) {
        ERROR("No access zones configured");
    }
}

void PeopleTracker::set_access_zones(const std::vector<TypedZone>& zones) {
    access_zones_ = zones;
}

void PeopleTracker::update(const std::vector<Detection>& filtered_detections, int frame_height) {
    set_access_zones(config_->GetTypedZones());
    if (access_zones_.empty()) {
        ERROR("Cannot update tracker: no access zones configured");
        return;
    }

    std::vector<Detection> high_conf_detections;
    std::vector<Detection> low_conf_detections;
    std::vector<bool> track_matched(people_.size(), false);
    std::vector<bool> det_matched(filtered_detections.size(), false);

    const float conf_threshold_high = 0.6f;
    const float conf_threshold_low = 0.2f;
    const float iou_threshold_high = 0.3f;
    const float iou_threshold_low = 0.4f;

    // Categorize detections by confidence
    for (size_t i = 0; i < filtered_detections.size(); ++i) {
        if (filtered_detections[i].confidence > conf_threshold_high) {
            high_conf_detections.push_back(filtered_detections[i]);
        } else if (filtered_detections[i].confidence > conf_threshold_low) {
            low_conf_detections.push_back(filtered_detections[i]);
        }
    }

    // Match high-confidence detections
    for (size_t t = 0; t < people_.size(); ++t) {
        float highest_iou = 0.0f;
        int best_det_idx = -1;

        for (size_t d = 0; d < high_conf_detections.size(); ++d) {
            if (det_matched[d]) continue;
            float iou = computeIoU(people_[t].getBoundingBox(), high_conf_detections[d].bounding_box);
            if (iou > highest_iou && iou > iou_threshold_high) {
                highest_iou = iou;
                best_det_idx = d;
            }
        }

        if (best_det_idx >= 0) {
            Position pos = {
                high_conf_detections[best_det_idx].bounding_box.x + high_conf_detections[best_det_idx].bounding_box.width / 2,
                high_conf_detections[best_det_idx].bounding_box.y + high_conf_detections[best_det_idx].bounding_box.height / 2
            };
            BoxSize size = {
                high_conf_detections[best_det_idx].bounding_box.width,
                high_conf_detections[best_det_idx].bounding_box.height
            };
            people_[t].update(pos, size, high_conf_detections[best_det_idx].confidence, access_zones_);
            track_matched[t] = true;
            det_matched[best_det_idx] = true;
            detect_movements(people_[t], frame_height);
        }
    }

    // Match low-confidence detections
    for (size_t t = 0; t < people_.size(); ++t) {
        if (track_matched[t]) continue;

        float highest_iou = 0.0f;
        int best_det_idx = -1;

        for (size_t d = 0; d < low_conf_detections.size(); ++d) {
            if (det_matched[d + high_conf_detections.size()]) continue;
            float iou = computeIoU(people_[t].getBoundingBox(), low_conf_detections[d].bounding_box);
            if (iou > highest_iou && iou > iou_threshold_low) {
                highest_iou = iou;
                best_det_idx = d;
            }
        }

        if (best_det_idx >= 0) {
            Position pos = {
                low_conf_detections[best_det_idx].bounding_box.x + low_conf_detections[best_det_idx].bounding_box.width / 2,
                low_conf_detections[best_det_idx].bounding_box.y + low_conf_detections[best_det_idx].bounding_box.height / 2
            };
            BoxSize size = {
                low_conf_detections[best_det_idx].bounding_box.width,
                low_conf_detections[best_det_idx].bounding_box.height
            };
            people_[t].update(pos, size, low_conf_detections[best_det_idx].confidence, access_zones_);
            track_matched[t] = true;
            det_matched[best_det_idx + high_conf_detections.size()] = true;
            detect_movements(people_[t], frame_height);
        }
    }

    // Update or remove unmatched tracks
    std::vector<TrackedPerson> new_people;
    for (size_t t = 0; t < people_.size(); ++t) {
        if (!track_matched[t]) {
            people_[t].missing_frames_++;
            if (people_[t].missing_frames_ < max_missing_frames_) {
                new_people.push_back(people_[t]);
            } else {
                detect_movements(people_[t], frame_height, true);
            }
        } else {
            new_people.push_back(people_[t]);
        }
    }

    // Add new tracks for unmatched high-confidence detections
    for (size_t d = 0; d < high_conf_detections.size(); ++d) {
        if (!det_matched[d]) {
            Position pos = {
                high_conf_detections[d].bounding_box.x + high_conf_detections[d].bounding_box.width / 2,
                high_conf_detections[d].bounding_box.y + high_conf_detections[d].bounding_box.height / 2
            };
            BoxSize size = {
                high_conf_detections[d].bounding_box.width,
                high_conf_detections[d].bounding_box.height
            };
            TrackedPerson new_person(next_id_++, pos, size, high_conf_detections[d].confidence,
                                    frame_height, access_zones_);
            new_people.push_back(new_person);
            detect_movements(new_person, frame_height);
        }
    }

    people_ = new_people;
}

void PeopleTracker::detect_movements(TrackedPerson& person, int frame_height, bool person_removed) {
    if (person.postion_history_.size() < 2) {
        return;
    }
    // Check if the person is moving from exit zone to entry zone or vice versa
    if (access_zones_.size() == 2) {
        if(person.first_zone_type_ == ZoneType::ENTRY && person.last_zone_type_ == ZoneType::EXIT && person_removed) {
            movement_callback_(person, "exited");
        } else if (person.first_zone_type_ == ZoneType::EXIT && person.last_zone_type_ == ZoneType::ENTRY && person_removed) {
            movement_callback_(person, "entered");

        }
    }
}

void PeopleTracker::draw(cv::Mat& frame) {
    if (access_zones_.empty()) {
        ERROR("Cannot draw: no access zones configured");
        return;
    }

    // Draw access zones
    for (size_t i = 0; i < access_zones_.size(); ++i) {
        const auto& zone = access_zones_[i].zone;
        cv::Scalar zone_color = access_zones_[i].type == ZoneType::ENTRY ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(frame, cv::Point(zone.x1, zone.y1), cv::Point(zone.x2, zone.y2), zone_color, 2);
        cv::putText(frame, access_zones_[i].getTypeString() + " Zone " + std::to_string(i),
                    cv::Point(zone.x1, zone.y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2);
    }

    // Draw tracked people
    for (const auto& person : people_) {
        cv::Rect bbox = person.getBoundingBox();
        cv::rectangle(frame, bbox, person.color_, 2);

        std::string label = "ID: " + std::to_string(person.id_) + " Conf: " + std::to_string(person.confidence_);
        if (person.last_zone_type_ != ZoneType::NONE) {
            // Fix string concatenation using std::string
            std::string zoneType = (person.last_zone_type_ == ZoneType::ENTRY) ? "Entry" : "Exit";
            label += std::string(" (") + zoneType + ")";
        }
        if (person.first_zone_type_ != ZoneType::NONE) {
            // Fix string concatenation using std::string
            std::string zoneType = (person.first_zone_type_ == ZoneType::ENTRY) ? "Entry" : "Exit";
            label += std::string(" (Spawned in ") + zoneType + ")";
        }
        int y = std::max(bbox.y - 10, 15);
        cv::putText(frame, label, cv::Point(bbox.x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, person.color_, 2);

        if (person.postion_history_.size() > 1) {
            for (size_t i = 1; i < person.postion_history_.size(); ++i) {
                float alpha = static_cast<float>(i) / person.postion_history_.size();
                cv::Scalar trail_color = person.color_ * alpha;
                cv::line(frame, cv::Point(person.postion_history_[i-1].x, person.postion_history_[i-1].y),
                         cv::Point(person.postion_history_[i].x, person.postion_history_[i].y), trail_color, 1);
                cv::circle(frame, cv::Point(person.postion_history_[i].x, person.postion_history_[i].y), 2, trail_color, -1);
            }
        }
    }
}

const std::vector<TrackedPerson>& PeopleTracker::get_tracked_people() const {
    return people_;
}

void PeopleTracker::set_movement_callback(MovementCallback callback) {
    movement_callback_ = callback;
}

}  // namespace tracker
