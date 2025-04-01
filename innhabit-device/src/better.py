import cv2

# Open the camera (0 refers to the default camera)
cap = cv2.VideoCapture(0)  # Change to your camera index (e.g., /dev/video0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set video resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't grab frame.")
        break

    # Display the frame
    cv2.imshow("Live Feed", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close window
cap.release()
cv2.destroyAllWindows()
