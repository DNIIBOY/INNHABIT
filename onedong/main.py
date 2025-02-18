import cv2
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from random import randint

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels (COCO dataset labels)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open the laptop cameraQ (use 0 for default webcam)
cap = cv2.VideoCapture(0)


@dataclass
class Position:
    x: int
    y: int

    @property
    def xy(self):
        return (self.x, self.y)


@dataclass
class Size:
    width: int
    height: int


@dataclass
class BoundingBox:
    position: Position
    size: Size

    @property
    def xy(self):
        return (
            self.position.x,
            self.position.y
        )

    @property
    def end_xy(self):
        return (
            self.position.x + self.size.width,
            self.position.y + self.size.height
        )


class Person:
    def __init__(
        self,
        bbox: BoundingBox,
    ) -> None:
        self.bbox = bbox
        self.position = None
        self.historical_positions = []
        self.color = (randint(50, 255), randint(50, 255), randint(50, 255))
        self.update(bbox)

    def update(self, bbox: BoundingBox):
        self.bbox = bbox
        self.position = Position(
            x=bbox.position.x + bbox.size.width // 2,
            y=bbox.position.y + bbox.size.height // 2,
        )
        self.historical_positions.append(self.position)


class PersonTracker:
    def __init__(self) -> None:
        self.people: dict[int, Person] = {}

    def get_distance(self, person1: Person, person2: Person) -> float:
        return ((person1.position.x - person2.position.x) ** 2 + (person1.position.y - person2.position.y) ** 2) ** 0.5

    def update(self, indexes, boxes) -> None:
        prev_people = deepcopy(self.people)
        self.people = {}
        if not len(indexes):
            return
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            bbox = BoundingBox(
                position=Position(x=x, y=y),
                size=Size(width=w, height=h),
            )
            person = Person(bbox)
            print(person)
            for person_id, prev_person in prev_people.items():
                if self.get_distance(person, prev_person) < 100:
                    self.people[person_id] = prev_person
                    self.people[person_id].update(bbox)
                    del prev_people[person_id]
                    break
            else:
                self.people[randint(0, 10000)] = person


def main():
    person_tracker = PersonTracker()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the width and height of the frame
        height, width, channels = frame.shape

        # Convert the frame to a blob (required by YOLO)
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Post-process the detections
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == "person":  # Detect humans
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = center_x - w // 2
                    y = center_y - h // 2

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maxima suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw the bounding boxes for detected humans
        person_tracker.update(indexes, boxes)
        for person in person_tracker.people.values():
            cv2.rectangle(frame, person.bbox.xy, person.bbox.end_xy, person.color, 2)
            for point in person.historical_positions:
                cv2.circle(frame, point.xy, 2, person.color, -1)

        # Show the output frame
        cv2.imshow("Human Detection", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
