/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "postprocess.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <set>
#include <vector>


/*-------------------------------------------
                  COCO labels
-------------------------------------------*/
static const char* labels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

const int anchor0[6] = {10, 13, 16, 30, 33, 23};
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
const int anchor2[6] = {116, 90, 156, 198, 373, 326};

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order, int filterId, float threshold) {
    for (int i = 0; i < validCount; ++i) {
        if (order[i] == -1 || classIds[i] != filterId) {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId) {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

template<typename T>
static int process(T *input, int grid_h, int grid_w, int *anchor, int height, int width, int stride,
                   std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                   int32_t zp, float scale, bool is_quantized) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    T thres = is_quantized ? qnt_f32_to_affine(threshold, zp, scale) : threshold;
#ifdef DEBUG
    printf("Processing grid: %dx%d, stride=%d, threshold=%f (quantized=%d), zp=%d, scale=%f\n",
           grid_h, grid_w, stride, threshold, is_quantized, zp, scale);
#endif
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                if (offset >= grid_len * PROP_BOX_SIZE * 3) continue; // Bounds check
                T box_confidence = input[offset + 4 * grid_len];
                float box_conf_f32 = is_quantized ? deqnt_affine_to_f32(box_confidence, zp, scale) : box_confidence;
#ifdef DEBUG
                if (a == 0 && i == 0 && j == 0) {
                    printf("Anchor %d, pos (%d,%d): box_confidence=%f (raw=%d), threshold=%f\n",
                           a, i, j, box_conf_f32, (int)box_confidence, is_quantized ? deqnt_affine_to_f32(thres, zp, scale) : thres);
                }
#endif
                if (box_conf_f32 >= (is_quantized ? deqnt_affine_to_f32(thres, zp, scale) : thres)) {
                    T *in_ptr = input + offset;
                    float box_x = (is_quantized ? deqnt_affine_to_f32(*in_ptr, zp, scale) : *in_ptr) * 2.0 - 0.5;
                    float box_y = (is_quantized ? deqnt_affine_to_f32(in_ptr[grid_len], zp, scale) : in_ptr[grid_len]) * 2.0 - 0.5;
                    float box_w = (is_quantized ? deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale) : in_ptr[2 * grid_len]) * 2.0;
                    float box_h = (is_quantized ? deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale) : in_ptr[3 * grid_len]) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    T maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
                        T prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    float obj_prob_f32 = (is_quantized ? deqnt_affine_to_f32(maxClassProbs, zp, scale) : maxClassProbs) * box_conf_f32;

                    if (obj_prob_f32 > threshold) {
                        objProbs.push_back(obj_prob_f32);
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

// Quantized implementation (RKNN)
int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t> &qnt_zps,
                 std::vector<float> &qnt_scales, detect_result_group_t *group, bool is_quantized) {
        
    memset(group, 0, sizeof(detect_result_group_t));

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    // Stride 8 (80x80 for 640x640, adjust for input size)
    int stride0 = 8;
    int grid_h0 = model_in_h / stride0;
    int grid_w0 = model_in_w / stride0;
    int validCount0 = process(input0, grid_h0, grid_w0, (int *)anchor0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
                              classId, conf_threshold, qnt_zps[0], qnt_scales[0], is_quantized);

    // Stride 16 (40x40 for 640x640)
    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    int validCount1 = process(input1, grid_h1, grid_w1, (int *)anchor1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
                              classId, conf_threshold, qnt_zps[1], qnt_scales[1], is_quantized);

    // Stride 32 (20x20 for 640x640)
    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    int validCount2 = process(input2, grid_h2, grid_w2, (int *)anchor2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
                              classId, conf_threshold, qnt_zps[2], qnt_scales[2], is_quantized);

    int validCount = validCount0 + validCount1 + validCount2;
    if (validCount <= 0) {
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }

    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));
    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    group->count = 0;
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) continue;

        int n = indexArray[i];
        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        group->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);
        group->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);
        group->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
        group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
        group->results[last_count].prop = obj_conf;
        const char *label = labels[id];
        strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

        last_count++;
    }
    group->count = last_count;

    return 0;
}

// Unquantized implementation (CPU/Jetson)
int post_process(float *input0, float *input1, float *input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t> &qnt_zps,
                 std::vector<float> &qnt_scales, detect_result_group_t *group, bool is_quantized) {
    return post_process(input0, input1, input2, model_in_h, model_in_w, conf_threshold, nms_threshold, scale_w, scale_h,
                        qnt_zps, qnt_scales, group, false);
}

// Instantiate templates
template int process<int8_t>(int8_t*, int, int, int*, int, int, int, std::vector<float>&, std::vector<float>&, std::vector<int>&, float, int32_t, float, bool);
template int process<float>(float*, int, int, int*, int, int, int, std::vector<float>&, std::vector<float>&, std::vector<int>&, float, int32_t, float, bool);