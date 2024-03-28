#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>

class errorPoint
{
private:
    cv::Point2f point;
    double distance;

public:
    errorPoint(cv::Point2f point, double distance) {
        this->point = point;
        this->distance = distance;
    }

    cv::Point2f getPoint() {
        return point;
    }

    double getDistance() {
        return distance;
    }
};

enum class ProcessingType {
    Full,                // Bilateral filter + Threshold + Canny
    ThresholdHigh,       // Chỉ Threshold với ngưỡng cao
    ThresholdMedium,     // Chỉ Threshold với ngưỡng trung bình
    ThresholdLow,        // Chỉ Threshold với ngưỡng thấp
    Canny                // Chỉ Canny
};

// Prototype của hàm processing_image với thêm tham số lựa chọn loại xử lý
void processing_image(const cv::Mat& image, cv::Mat& bin_img, cv::Mat& blurred_image, cv::Mat& canny_edges, ProcessingType type = ProcessingType::Full);

std::vector<cv::Point> extract_contour(const cv::Mat& image, ProcessingType type = ProcessingType::Full) {
    cv::Mat bin_img, blurred_image, canny_edges;

    // Gọi hàm processing_image với loại xử lý mong muốn
    processing_image(image, bin_img, blurred_image, canny_edges, type);

    // Sử dụng hàm findContours để tìm contours từ ảnh cạnh
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny_edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    // Loại bỏ contours có độ dài nhỏ hơn 100
    contours.erase(std::remove_if(contours.begin(), contours.end(), [](const std::vector<cv::Point>& cnt) {
        return cv::arcLength(cnt, true) <= 100;
        }), contours.end());

    // Nối các contours thành một vector chứa các điểm
    std::vector<cv::Point> contour;
    for (const auto& cnt : contours) {
        contour.insert(contour.end(), cnt.begin(), cnt.end());
    }
    return contour;
}

void processing_image(const cv::Mat& image, cv::Mat& bin_img, cv::Mat& blurred_image, cv::Mat& canny_edges, ProcessingType type) {
    switch (type) {
    case ProcessingType::Full:
    {
        // Áp dụng lọc Bilateral để làm mờ ảnh
        cv::bilateralFilter(image, blurred_image, 11, 225, 75);
        // Chuyển ảnh đã làm mờ sang ảnh nhị phân
        cv::threshold(blurred_image, bin_img, 175, 255, cv::THRESH_BINARY);
        // Sử dụng Canny để tìm cạnh từ ảnh nhị phân
        cv::Canny(bin_img, canny_edges, 125, 215);
    }
    break;
    case ProcessingType::ThresholdHigh:
        cv::threshold(image, bin_img, 200, 255, cv::THRESH_BINARY); // Ngưỡng cao
        blurred_image = image.clone();
        canny_edges = bin_img.clone();
        break;
    case ProcessingType::ThresholdMedium:
        cv::threshold(image, bin_img, 128, 255, cv::THRESH_BINARY); // Ngưỡng trung bình
        blurred_image = image.clone();
        canny_edges = bin_img.clone();
        break;
    case ProcessingType::ThresholdLow:
        cv::threshold(image, bin_img, 50, 255, cv::THRESH_BINARY); // Ngưỡng thấp
        blurred_image = image.clone();
        canny_edges = bin_img.clone();
        break;
    case ProcessingType::Canny:
    {
        // Chỉ áp dụng Canny để tìm cạnh
        cv::Canny(image, canny_edges, 125, 215);
        blurred_image = image.clone(); // Giữ nguyên ảnh gốc cho blurred_image
        bin_img = image.clone(); // Giữ nguyên ảnh gốc cho bin_img
    }
    break;
    default:
        // Nếu không có loại xử lý nào được chọn, trả về ảnh gốc
        blurred_image = image.clone();
        bin_img = image.clone();
        canny_edges = image.clone();
        break;
    }
}
std::map<std::pair<int, int>, std::vector<int>>
devide_bin(std::vector<cv::Point> contour, int stride) {
    // Divide points in contour into bins
    std::map<std::pair<int, int>, std::vector<int>> bin;
    for (int idx = 0; idx < contour.size(); ++idx) {
        auto& point = contour[idx];
        bin[std::make_pair(point.x / stride, point.y / stride)].push_back(idx);
    }
    return bin;
}

// Define trainTemplate function:
std::pair< std::vector<cv::Point>, std::vector<int> >
trainTemplate(const cv::Mat& temp_img) {
    std::vector<cv::Point> temp_contour = extract_contour(temp_img);
    int height = temp_img.rows;
    int width = temp_img.cols;

    return { temp_contour, {width, height} };
}

// Convert coordinate for error point:
cv::Point convert_coor(cv::Point point, cv::Mat convert_mat) {
    int x, y, x_, y_;
    x = point.x;
    y = point.y;
    // Formula:
    x_ = round(x * convert_mat.at<double>(0, 0) + y * convert_mat.at<double>(0, 1) + convert_mat.at<double>(0, 2));
    y_ = round(x * convert_mat.at<double>(1, 0) + y * convert_mat.at<double>(1, 1) + convert_mat.at<double>(1, 2));
    return cv::Point(x_, y_);
}

// Define compareContour function:
std::pair<std::vector<errorPoint>, std::vector<errorPoint>>
compareContour(const std::vector<cv::Point>& temp_contour, std::vector<int> temp_size, const cv::Mat& targ_img, std::vector<cv::Point2i> rect, int distance_thresh) {
    auto start = std::chrono::high_resolution_clock::now();
    int width = temp_size[0] - 1, height = temp_size[1] - 1;
    std::vector<cv::Point2f> src = { cv::Point2f(0, 0), cv::Point2f(width, 0), cv::Point2f(0, height) };
    std::vector<cv::Point2f> dst = { cv::Point2f(rect[0]), cv::Point2f(rect[1]), cv::Point2f(rect[3]) };
    cv::Mat convert_mat = cv::getAffineTransform(src, dst);

    // Convert coor of template contour:
    std::vector<cv::Point> converted_temp_contour;
    for (const cv::Point& point : temp_contour) {
        converted_temp_contour.push_back(cv::Point2f(convert_coor(point, convert_mat)));
    }

    // Divide bin:
    int patch_size = 2 * distance_thresh;
    std::vector<cv::Point> targ_contour = extract_contour(targ_img);
    std::map < std::pair<int, int>, std::vector<int> > temp_bin = devide_bin(converted_temp_contour, patch_size),
        targ_bin = devide_bin(targ_contour, patch_size);

    //Calculate distance and give defect position
    std::vector<errorPoint> targ_output, temp_output;

    // Calculate distance from target contour to template contour
    for (const cv::Point& point : targ_contour) {
        int x = point.x / patch_size, y = point.y / patch_size;   // Lẩy chỉ số của bin chứa điểm đang xét
        int k1 = (point.x % patch_size >= distance_thresh) ? 1 : -1, k2 = (point.y % patch_size >= distance_thresh) ? 1 : -1; // Lấy chỉ số của 3 bin lân cận cần xét

        std::vector<std::pair<int, int>> key_vec{ {x, y}, {x + k1, y}, {x, y + k2}, {x + k1, y + k2} };
        // Tạo list gồm các index trong các bin
        std::vector<int> idx_list;
        for (const auto& k : key_vec) {
            if (temp_bin.count(k) > 0) {
                idx_list.insert(idx_list.end(), temp_bin[k].begin(), temp_bin[k].end());
            }
        }

        if (idx_list.size() != 0) {     // Nếu list không rỗng, ta tính toán với các điểm trong list
            double min_distance = std::numeric_limits<double>::max();

            for (const auto& idx : idx_list) {
                double distance = std::abs(converted_temp_contour[idx].x - point.x) + std::abs(converted_temp_contour[idx].y - point.y);
                min_distance = std::min(min_distance, distance);
            }
            if (min_distance > distance_thresh) {
                errorPoint ER(cv::Point2f(point), min_distance);
                targ_output.push_back(ER);
            }
        }
        else {        // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
            errorPoint ER(cv::Point2f(point), distance_thresh + 1);
            targ_output.push_back(ER);
        }
    }

    // Calculate distance from coverted template contour to target contour
    for (const cv::Point& point : converted_temp_contour) {
        int x = point.x / patch_size, y = point.y / patch_size;
        int k1 = (point.x % patch_size >= distance_thresh) ? 1 : -1, k2 = (point.y % patch_size >= distance_thresh) ? 1 : -1;

        std::vector<std::pair<int, int>> key_vec{ {x, y}, {x + k1, y}, {x, y + k2}, {x + k1, y + k2} };
        std::vector<int> idx_list;
        for (const auto& k : key_vec) {
            if (targ_bin.count(k) > 0) {
                idx_list.insert(idx_list.end(), targ_bin[k].begin(), targ_bin[k].end());
            }
        }

        if (idx_list.size() != 0) {
            double min_distance = std::numeric_limits<double>::max();
            for (const auto& idx : idx_list) {
                double distance = std::abs(targ_contour[idx].x - point.x) + std::abs(targ_contour[idx].y - point.y);
                min_distance = std::min(min_distance, distance);
            }
            if (min_distance > distance_thresh) {
                errorPoint ER(cv::Point2f(point), min_distance);
                targ_output.push_back(ER);
            }
        }
        else {        // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
            errorPoint ER(cv::Point2f(point), distance_thresh + 1);
            targ_output.push_back(ER);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << std::endl;

    return { targ_output, temp_output };
}