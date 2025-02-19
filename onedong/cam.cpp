#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

int main() {
    // Open the camera using CAP_V4L2 (Video4Linux2) API
    VideoCapture cap(0, CAP_V4L2);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video stream" << endl;
        return -1;
    }

    // Set camera properties (optional, similar to Python code)
    // cap.set(CAP_PROP_FRAME_WIDTH, 1280);  // Set width to 1280
    // cap.set(CAP_PROP_FRAME_HEIGHT, 720); // Set height to 720
    // cap.set(CAP_PROP_FPS, 30);           // Set FPS to 30

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Could not capture frame" << endl;
            break;
        }

        // Display the frame
        imshow("Webcam Output", frame);
        if (waitKey(1) == 27) { // Exit on 'ESC'
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
