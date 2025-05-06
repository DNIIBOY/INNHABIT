#ifndef ZONE_TYPE_H
#define ZONE_TYPE_H

#include <string>
#include <vector>

namespace tracker {

/**
 * @brief Represents a rectangular bounding box zone.
 */
struct BoxZone {
    int x1, y1; ///< Top-left corner coordinates.
    int x2, y2; ///< Bottom-right corner coordinates.
};

/**
 * @brief Enum representing the type of zone (entry, exit, or none).
 */
enum class ZoneType {
    ENTRY,  ///< Zone where people enter.
    EXIT,   ///< Zone where people exit.
    NONE    ///< No specific zone type.
};

/**
 * @brief Pairs a BoxZone with its type (entry or exit).
 */
struct TypedZone {
    BoxZone zone;      ///< The zone's coordinates.
    ZoneType type;     ///< The type of zone (entry or exit).

    /**
     * @brief Constructs a TypedZone with coordinates and type.
     * @param x1 Top-left x coordinate.
     * @param y1 Top-left y coordinate.
     * @param x2 Bottom-right x coordinate.
     * @param y2 Bottom-right y coordinate.
     * @param zone_type The type of zone (entry or exit).
     */
    TypedZone(int x1, int y1, int x2, int y2, ZoneType zone_type)
        : zone{x1, y1, x2, y2}, type(zone_type) {}

    /**
     * @brief Constructs a TypedZone from an existing BoxZone.
     * @param box_zone The BoxZone to use.
     * @param zone_type The type of zone (entry or exit).
     */
    TypedZone(const BoxZone& box_zone, ZoneType zone_type)
        : zone(box_zone), type(zone_type) {}

    /**
     * @brief Checks if a point is within this zone.
     * @param x X coordinate to check.
     * @param y Y coordinate to check.
     * @return True if the point is inside the zone, false otherwise.
     */
    bool containsPoint(int x, int y) const {
        return x >= zone.x1 && x <= zone.x2 && y >= zone.y1 && y <= zone.y2;
    }

    /**
     * @brief Gets a string representation of the zone type.
     * @return "Entry", "Exit", or "None" depending on the zone type.
     */
    std::string getTypeString() const {
        switch (type) {
            case ZoneType::ENTRY: return "Entry";
            case ZoneType::EXIT: return "Exit";
            default: return "None";
        }
    }
};

}  // namespace tracker

#endif  // ZONE_TYPE_H