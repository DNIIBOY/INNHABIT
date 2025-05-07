#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/**
 * @struct Detection
 * @brief Structure used for storing detection results.
 */
struct Detection {
    string class_id;         ///< Class label of the detected object
    float confidence;        ///< Confidence score of the detection
    Rect bounding_box;       ///< Bounding box coordinates
};

/**
 * @struct DetectionOutput
 * @brief Structure used for storing platform-specific inference outputs.
 */
struct DetectionOutput {
    vector<void*> buffers;   ///< Pointers to output buffers
    vector<cv::Mat> outputs;     ///< For OpenCV DNN outputs
    
    /**
     * @brief Clears all output buffers.
     */
    void clear() const {
        const_cast<vector<void*>&>(buffers).clear();
        const_cast<vector<Mat>&>(outputs).clear();
    }
};

/**
 * @class GenericDetector
 * @brief Base class for object detection with common functionality.
 */
class GenericDetector {
protected:
    vector<string> target_classes_;  ///< List of target class labels
    bool initialized_ = false;       ///< Flag indicating if the detector is initialized
    vector<Detection> detections_;   ///< Stores detected objects

    /**
     * @brief Initializes the detector with the given model.
     * @param model_path Path to the model file.
     */
    virtual void initialize(const string& model_path) = 0;

    /**
     * @brief Releases any inference-related outputs.
     */
    virtual void releaseOutputs() = 0;

    /**
     * @brief Clamps a value within the given range.
     * @param value The input value.
     * @param min_value Minimum allowed value.
     * @param max_value Maximum allowed value.
     * @return The clamped value.
     */
    int clamp(int value, int min_value, int max_value);

    /**
     * @brief Preprocesses the input frame before inference.
     * @param frame The input frame.
     */
    virtual void preprocess(cv::Mat& frame) = 0;

    /**
     * @brief Runs inference on the input frame.
     * @param frame The input frame.
     */
    virtual void inference(cv::Mat& frame) = 0;

    /**
     * @brief Processes the inference results to extract detections.
     * @param frame The input frame.
     */
    virtual void postprocess(cv::Mat& frame) = 0;

public:
    /**
     * @brief Constructor for GenericDetector.
     * @param model_path Path to the detection model.
     * @param target_classes List of target classes for detection.
     */
    GenericDetector(const string& model_path, const vector<string>& target_classes);

    /**
     * @brief Virtual destructor.
     */
    virtual ~GenericDetector() = default;

    /**
     * @brief Runs the detection pipeline on the input frame.
     * @param frame The input frame.
     */
    virtual void detect(cv::Mat& frame);

    /**
     * @brief Gets the detected objects.
     * @return A reference to the vector of detections.
     */
    const vector<Detection>& getDetections() const { return detections_; }

    /**
     * @brief Checks if the detector has been properly initialized.
     * @return True if initialized, otherwise false.
     */
    bool isInitialized() const { return initialized_; }
};

/**
 * @brief Factory function to create an appropriate detector.
 * @param model_path Path to the model file.
 * @param target_classes List of target classes for detection.
 * @return A pointer to a GenericDetector instance.
 */
GenericDetector* createDetector(const string& model_path, const vector<string>& target_classes);

#endif // DETECTOR_H
