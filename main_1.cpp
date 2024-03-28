#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/flann.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <sys/stat.h>

#include "function.cpp"
#include "ROI_Cpp/roi.cpp"

namespace fs = std::filesystem;

std::pair < std::vector < int >, double >
makeROI(const cv::Mat& temp_img, const cv::Mat& targ_img) {
    cv::Ptr<cv::SiftFeatureDetector> detector;
    detector = cv::SiftFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints_temp, keypoints_targ;
    cv::Mat descriptors_temp, descriptors_targ;
    cv::BFMatcher bf;

    detector->detectAndCompute(temp_img, cv::noArray(), keypoints_temp, descriptors_temp);
    detector->detectAndCompute(targ_img, cv::noArray(), keypoints_targ, descriptors_targ);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    bf.knnMatch(descriptors_temp, descriptors_targ, knn_matches, 2);
    std::vector<cv::DMatch> good_matches;
    for (const auto& match_pair : knn_matches) {
        if (match_pair[0].distance < 0.75 * match_pair[1].distance) {
            good_matches.push_back(match_pair[0]);
        }
    }
    std::vector<cv::Point2f> points_temp, points_targ;
    for (const auto& m : good_matches) {
        points_temp.push_back(keypoints_temp[m.queryIdx].pt);
        points_targ.push_back(keypoints_targ[m.trainIdx].pt);
    }

    // Ước lượng phép biến đổi Affine
    cv::Mat M = cv::estimateAffinePartial2D(points_temp, points_targ);
    // Trích xuất các giá trị dịch chuyển và góc xoay
    int Dx = round(M.at<double>(0, 2)); //truc 0x
    int Dy = round(M.at<double>(1, 2)); //truc Oy
    double Dr = std::atan(M.at<double>(1, 0) / M.at<double>(0, 0));
    if (abs(cos(Dr) - M.at<double>(0, 0)) < 0.01) { Dr = Dr * 180 / CV_PI; }
    else { Dr = 180 + Dr * 180 / CV_PI; }

    std::vector<int> rect{ Dx, Dy };
    return { rect, Dr };
}

std::vector<cv::Point2i>
get4POint(int x, int y, int w, int h, double angle) {
    std::vector<cv::Point2i> vts{
            Point2i(x, y),
            Point2i(x + w, y),
            Point2i(x + w, y + h),
            Point2i(x, y + h)
    };
    if (angle == 0) { return vts; };
    double angle_rad = angle * CV_PI / 180;
    double alpha_w = std::atan(w * 1.0 / h);
    double alpha_h = std::atan(h * 1.0 / w);
    double half_cross = sqrt(w * w + h * h) / 2;
    cv::Point2f center = cv::Point2f(x + half_cross * sin(alpha_w - angle_rad), y + half_cross * cos(alpha_w - angle_rad));
    vts[1] = cv::Point2i(round(center.x + half_cross * cos(alpha_h - angle_rad)), round(center.y - half_cross * sin(alpha_h - angle_rad)));
    vts[2] = cv::Point2i(round(2 * center.x - vts[0].x), round(2 * center.y - vts[0].y));
    vts[3] = cv::Point2i(round(2 * center.x - vts[1].x), round(2 * center.y - vts[1].y));
    return vts;
}


int main() {
    std::cout << "Select processing method:\n";
    std::cout << "1. Full\n";
    std::cout << "2. Canny\n";
    std::cout << "3. ThresholdHigh\n";
    std::cout << "4.ThresholdMedium\n ";
    std::cout << "5.ThresholdLow\n ";
    int choice;
    std::cin >> choice;

    ProcessingType type;
    switch (choice) {
    case 1:
        type = ProcessingType::Full;
        break;
    case 2:
        type = ProcessingType::Canny;
        break;
    case 3:
        type = ProcessingType::ThresholdHigh;
        break;
    case 4:
        type = ProcessingType::ThresholdMedium;
        break;
    case 5:
        type = ProcessingType::ThresholdLow;
        break;
    default:
        std::cout << "Invalid choice. Using Full as default.\n";
        type = ProcessingType::Full;
    }



    //double threshold = 215;
    int distance_thresh = 15;
    //double min_length_contour = 100;

    std::string path = "C:\\C++ Project\\Inspect_contours\\Image\\front";
    //std::cout<< "Enter the path of imageset: ";
    //std::cin >> path;

    // This structure would distinguish a file from a directory
    struct stat sb;
    std::vector<std::string> list_image;
    for (const auto& entry : fs::directory_iterator(path)) {
        // Converting the path to const char * in the subsequent lines
        std::filesystem::path outfilename = entry.path();
        std::string outfilename_str = outfilename.string();
        const char* img_path = outfilename_str.c_str();

        // Testing whether the path points to a non-directory or not
        if (stat(img_path, &sb) == 0 && !(sb.st_mode & S_IFDIR)) {
            list_image.push_back(img_path);
        }
    }

    std::vector<cv::Mat> images;
    for (std::string name : list_image) {
        images.push_back(cv::imread(name, 0));
    };

    std::vector<cv::Point> temp_contour;
    std::map<std::pair<int, int>, std::vector<int>> temp_bin;
    std::vector<int> temp_size;
    tie(temp_contour, temp_size) = trainTemplate(images[15]);

    std::vector<std::vector<std::vector<errorPoint>>> pos_list;

    for (size_t i = 0; i < 6 /*list_image.size()*/; ++i) {
        std::vector<errorPoint> pos_1, pos_2;
        //RectangleRoi ROI(r[0], r[1], temp_size[0], temp_size[1], angle);
        //std::vector<cv::Point2d> rec = ROI.getVertices();
        std::vector < int > r;
        double angle;
        tie(r, angle) = makeROI(images[15], images[i]);
        std::vector<cv::Point2i> vts = get4POint(r[0], r[1], temp_size[0], temp_size[1], angle);

        tie(pos_1, pos_2) = compareContour(temp_contour, temp_size, images[i], vts, distance_thresh);
        pos_list.push_back({ pos_1, pos_2 });
    }

    std::cout << "Enter: ";
    int k;
    std::cin >> k;
    if (pos_list[k][0].empty() && pos_list[k][1].empty()) {
        std::cout << "No error" << "\n";
    }
    else {
        for (int i = 0; i < 2; i++) {
            for (errorPoint er : pos_list[k][i]) {
                std::cout << er.getPoint() << " - " << er.getDistance() << "\n";
            }
        }
    }
    //std::cout << pos_list[k][0] << pos_list[k][1] << "\n";

    while (k >= 0) {
        std::cin >> k;
        if (pos_list[k][0].empty() && pos_list[k][1].empty()) {
            std::cout << "No error" << "\n";
        }
        else {
            for (int i = 0; i < 2; i++) {
                for (errorPoint er : pos_list[k][i]) {
                    std::cout << er.getPoint() << " - " << er.getDistance() << "\n";
                }
            }
        }
        //std::cout << pos_list[k][0] << pos_list[k][1] << "\n";
    }

    return 0;
};