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


namespace fs = std::filesystem;


// Define function - extract contour and divide points into bins
std::pair<std::vector<cv::Point>, std::map<std::pair<int, int>, std::vector<int>>> // Format of output
process_contour(const cv::Mat& image, double threshold, int stride) {

    // Threshold image input and save into bin_img
    cv::Mat bin_img;
    cv::threshold(image, bin_img, threshold, 255, cv::THRESH_BINARY_INV);
    // Use function findContours in OpenCV to extract contours of bin_img
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    // Remove contours having length smaller than 100
    contours.erase(std::remove_if(contours.begin(), contours.end(), [](const std::vector<cv::Point>& cnt) {
        return cv::arcLength(cnt, true) <= 100;
        }), contours.end());

    // Concatenate contours (a vector of vectors containing points) into contour (a vector containing points)
    std::vector<cv::Point> contour;
    for (const auto& cnt : contours) {
        contour.insert(contour.end(), cnt.begin(), cnt.end());
    }

    // Divide points in contour into bins
    std::map<std::pair<int, int>, std::vector<int>> bin;
    for (int idx = 0; idx < contour.size(); ++idx) {
        auto& point = contour[idx];
        bin[std::make_pair(point.x / stride, point.y / stride)].push_back(idx);
    }
    return { contour, bin };
}

// Function extracting contour for KDTree
cv::Mat extractContour(const cv::Mat& image, double threshold) {
    // Threshold image input and save into bin_img
    cv::Mat bin_img;
    cv::threshold(image, bin_img, threshold, 255, cv::THRESH_BINARY_INV);

    // Use function findContours in OpenCV to extract contours of bin_img
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    // Remove contours having length smaller than 100
    contours.erase(std::remove_if(contours.begin(), contours.end(), [](const std::vector<cv::Point>& cnt) {
        return cv::arcLength(cnt, true) <= 100;
        }), contours.end());

    // Concatenate contours (a vector of vectors containing points) into contour (a vector containing points)
    std::vector<cv::Point2f> points;
    for (const auto& contour : contours) {
        for (const auto& point : contour) {
            points.push_back(cv::Point2f(point.x, point.y));
        }
    }
    return cv::Mat(points).reshape(1).clone();
}



class InspectContour {
private:
    // 2 attribute variables
    float threshold;
    short distance_thresh;
    cv::Ptr<cv::SiftFeatureDetector> detector; // detector for template matching

public:
    // Constructor
    InspectContour(float threshold, short distance_thresh) : distance_thresh(distance_thresh), threshold(threshold)
    {
        detector = cv::SiftFeatureDetector::create();
    }

    // Define trainTemplate function:
    std::tuple<std::vector<cv::KeyPoint>, cv::Mat, cv::Mat>
        trainTemplate(const cv::Mat& temp_img) {

        std::vector<cv::KeyPoint> keypoints_temp;
        cv::Mat descriptors_temp;
        detector->detectAndCompute(temp_img, cv::noArray(), keypoints_temp, descriptors_temp);
        return { keypoints_temp, descriptors_temp, temp_img };
    }

    // Define compareContour function:
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
        compareContour(const std::vector<cv::KeyPoint>& keypoints_temp, const cv::Mat& descriptors_temp, const cv::Mat& temp_img, const cv::Mat& targ_img) {

        // Template mactching:
        cv::BFMatcher bf;
        std::vector<cv::KeyPoint> keypoints_targ;
        cv::Mat descriptors_targ;
        detector->detectAndCompute(targ_img, cv::noArray(), keypoints_targ, descriptors_targ);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        bf.knnMatch(descriptors_temp, descriptors_targ, knn_matches, 2); // kNN
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
        short height = targ_img.rows;
        short width = targ_img.cols;
        cv::Mat M = cv::estimateAffinePartial2D(points_targ, points_temp);
        cv::Mat warp_img;
        cv::warpAffine(temp_img, warp_img, M, cv::Size(width, height), cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar(255));

        auto start = std::chrono::high_resolution_clock::now();

        // Find contour and divide points into bins
        short patch_size = 2 * distance_thresh;
        std::vector<cv::Point> warp_contour, targ_contour;
        std::map<std::pair<int, int>, std::vector<int>> warp_bin, targ_bin;
        tie(warp_contour, warp_bin) = process_contour(warp_img, threshold, patch_size);
        tie(targ_contour, targ_bin) = process_contour(targ_img, threshold, patch_size);

        // Calculate distance 1
        std::vector<double> distance_1(targ_contour.size());
        for (int i = 0; i < targ_contour.size(); i++) {
            const cv::Point& point = targ_contour[i];   // Chạy lần lượt các điểm thuộc targ_contour
            int x = point.x / patch_size;
            int y = point.y / patch_size;   // Lẩy chỉ số của bin chứa điểm đang xét
            // Lấy chỉ số của 3 bin lân cận cần xét
            int k1 = (point.x % patch_size >= distance_thresh) ? 1 : -1;
            int k2 = (point.y % patch_size >= distance_thresh) ? 1 : -1;
            std::vector<std::pair<int, int>> key_vec{ {x, y}, {x + k1, y}, {x, y + k2}, {x + k1, y + k2} };
            // Tạo list gồm các index trong các bin
            std::vector<int> idx_list;
            for (const auto& k : key_vec) {
                if (warp_bin.count(k) > 0) {
                    idx_list.insert(idx_list.end(), warp_bin[k].begin(), warp_bin[k].end());
                }
            }
            if (idx_list.size() != 0) {     // Nếu list không rỗng, ta tính toán với các điểm trong list
                double min_distance = std::numeric_limits<double>::max();
                for (const auto& idx : idx_list) {
                    double distance = std::abs(warp_contour[idx].x - point.x) + std::abs(warp_contour[idx].y - point.y);
                    min_distance = std::min(min_distance, distance);
                }
                distance_1[i] = min_distance;
            }
            else {        // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
                distance_1[i] = distance_thresh + 1;
            }
        }

        // Calculate distance 2
        std::vector<double> distance_2(warp_contour.size());
        for (int i = 0; i < warp_contour.size(); i++) {
            const cv::Point& point = warp_contour[i];
            int x = point.x / patch_size;
            int y = point.y / patch_size;
            int k1 = (point.x % patch_size >= distance_thresh) ? 1 : -1;
            int k2 = (point.y % patch_size >= distance_thresh) ? 1 : -1;
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
                distance_2[i] = min_distance;
            }
            else {
                distance_2[i] = distance_thresh + 1;
            }
        }

        // Calculate position distance greater than distance_thresh
        std::vector<cv::Point2f> pos_1, pos_2;
        for (int i = 0; i < distance_1.size(); i++) {
            if (distance_1[i] > distance_thresh) {
                pos_1.push_back(cv::Point2f(targ_contour[i]));
            }
        }
        for (int i = 0; i < distance_2.size(); i++) {
            if (distance_2[i] > distance_thresh) {
                pos_2.push_back(cv::Point2f(warp_contour[i]));
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << elapsed.count() << std::endl;

        return { pos_1, pos_2 };
    }


    // KD-Tree
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
        compareByKDTree(const std::vector<cv::KeyPoint>& keypoints_temp, const cv::Mat& descriptors_temp, const cv::Mat& temp_img, const cv::Mat& targ_img) {

        // Template mactching:
        cv::BFMatcher bf;
        std::vector<cv::KeyPoint> keypoints_targ;
        cv::Mat descriptors_targ;
        detector->detectAndCompute(targ_img, cv::noArray(), keypoints_targ, descriptors_targ);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        bf.knnMatch(descriptors_temp, descriptors_targ, knn_matches, 2); // kNN
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
        int height = targ_img.rows;
        int width = targ_img.cols;
        cv::Mat M = cv::estimateAffinePartial2D(points_targ, points_temp);
        cv::Mat warp_img;
        cv::warpAffine(temp_img, warp_img, M, cv::Size(width, height), cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar(255));

        auto start = std::chrono::high_resolution_clock::now();

        // Find contour
        cv::Mat warp_contour, targ_contour;
        warp_contour = extractContour(warp_img, threshold);
        targ_contour = extractContour(targ_img, threshold);

        std::vector<cv::Point2f> pos_1, pos_2;
        // Calculate distance 1
        cv::flann::Index flannIndex_targ(warp_contour, cv::flann::KDTreeIndexParams(2));
        cv::Mat targ_index(targ_contour.rows, 1, CV_32S), targ_dist(targ_contour.rows, 1, CV_32S);
        flannIndex_targ.knnSearch(targ_contour, targ_index, targ_dist, 1, cv::flann::SearchParams(64));
        for (int i = 0; i < targ_index.rows; i++) {
            if (sqrt(targ_dist.at<float>(i, 0)) > distance_thresh) {
                pos_1.push_back(targ_contour.at<cv::Point2f>(i, 0));
            }
        }

        // Calculate distance 2
        cv::flann::Index flannIndex_temp(targ_contour, cv::flann::KDTreeIndexParams(2));
        cv::Mat temp_index(warp_contour.rows, 1, CV_32S), temp_dist(warp_contour.rows, 1, CV_32S);
        flannIndex_temp.knnSearch(warp_contour, temp_index, temp_dist, 1, cv::flann::SearchParams(64));
        for (int i = 0; i < temp_index.rows; i++) {
            if (sqrt(temp_dist.at<float>(i, 0)) > distance_thresh) {
                pos_2.push_back(warp_contour.at<cv::Point2f>(i, 0));
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << elapsed.count() << std::endl;

        return { pos_1, pos_2 };
    }
};


int main() {
    std::string path = "C:\\Users\\Admin\\source\\repos\\InspectContour\\NG";
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

    std::cout << "Enter the index of the template image: ";
    int idxOfTemp;
    std::cin >> idxOfTemp;


    InspectContour IC(215, 15);
    std::vector<cv::KeyPoint> keypoints_temp;
    cv::Mat descriptors_temp;
    cv::Mat temp_img;
    tie(keypoints_temp, descriptors_temp, temp_img) = IC.trainTemplate(images[idxOfTemp]);


    std::vector<std::vector<std::vector<cv::Point2f>>> pos_list;
    for (size_t i = 0; i < list_image.size(); ++i) {
        if (i != idxOfTemp) {
            std::vector<cv::Point2f> pos_1, pos_2;
            tie(pos_1, pos_2) = IC.compareContour(keypoints_temp, descriptors_temp, temp_img, images[i]);
            //tie(pos_1, pos_2) = IC.compareByKDTree(keypoints_temp, descriptors_temp, temp_img, images[i]);

            pos_list.push_back({ pos_1, pos_2 });
        }
    }
    std::cout << "Enter: ";
    int k;
    std::cin >> k;
    std::cout << pos_list[k][0] << pos_list[k][1] << "\n";

    while (k >= 0) {
        std::cin >> k;
        std::cout << pos_list[k][0] << pos_list[k][1] << "\n";
    }

    return 0;
};




// Crop image from 4 corner points
/*
cv::Mat
four_point_transform(cv::Mat image, std::vector<cv::Point2d> rect) {
    cv::Point2f tl = cv::Point2f(rect[0]);
    cv::Point2f tr = cv::Point2f(rect[1]);
    cv::Point2f br = cv::Point2f(rect[2]);
    cv::Point2f bl = cv::Point2f(rect[3]);

    float widthA = std::sqrt(std::pow(br.x - bl.x, 2) + std::pow(br.y - bl.y, 2));
    float widthB = std::sqrt(std::pow(tr.x - tl.x, 2) + std::pow(tr.y - tl.y, 2));
    int maxWidth = std::max(static_cast<int>(widthA), static_cast<int>(widthB));

    float heightA = std::sqrt(std::pow(tr.x - br.x, 2) + std::pow(tr.y - br.y, 2));
    float heightB = std::sqrt(std::pow(tl.x - bl.x, 2) + std::pow(tl.y - bl.y, 2));
    int maxHeight = std::max(static_cast<int>(heightA), static_cast<int>(heightB));

    //cv::Mat dst = (cv::Mat_<int>(4, 2) << 0, 0, maxWidth - 1, 0, maxWidth - 1, maxHeight - 1, 0, maxHeight - 1);
    std::vector<cv::Point2f> dst = { cv::Point2f(0, 0), cv::Point2f(maxWidth - 1, 0), cv::Point2f(maxWidth - 1, maxHeight - 1), cv::Point2f(0, maxHeight - 1) };
    std::vector<cv::Point2f> new_rect = { tl, tr, br, bl };

    cv::Mat M = cv::getPerspectiveTransform(new_rect, dst);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(maxWidth, maxHeight));

    return warped;
}
*/

// Convert coor
/*
/// <summary>
/// Dịch chuyển điểm: xoay và dịch chuyển theo tọa độ tâm chỉ định
/// </summary>
/// <param name="points"></param>
/// <param name="dx"></param>
/// <param name="dy"></param>
/// <param name="angle"></param>
/// <param name="origin"></param>
/// <returns></returns>
public static Point2d[] EuclideanTransformPoints(IEnumerable<Point2d> points, double dx, double dy, double angle, Point origin)
{
    int n = points.Count();
    Point2d[] dstPoints = new Point2d[n];

    var sinPhi = Math.Sin(angle * Math.PI / 180);
    var cosPhi = Math.Cos(angle * Math.PI / 180);

    for (int i = 0; i < n; i++)
    {
        var p = points.ElementAt(i);
        // Tọa độ theo tâm xoay
        var x1 = p.X - origin.X;
        var y1 = p.Y - origin.Y;
        // xoay và dịch lại tâm cũ
        var x2 = origin.X + (x1 * cosPhi - y1 * sinPhi);
        var y2 = origin.Y + (x1 * sinPhi + y1 * cosPhi);
        dstPoints[i] = new Point2d(x2 + dx, y2 + dy);
    }
    return dstPoints;
}*/

//inv_convert_coor
/*
cv::Point inv_convert_coor(cv::Point point, cv::Mat convert_mat) {
    double a00 = convert_mat.at<double>(0, 0),
           a01 = convert_mat.at<double>(0, 1),
           a02 = convert_mat.at<double>(0, 2),
           a10 = convert_mat.at<double>(1, 0),
           a11 = convert_mat.at<double>(1, 1),
           a12 = convert_mat.at<double>(1, 2);
    float x, y;
    int x_, y_;
    x = point.x * 1.0 - a02;
    y = point.y * 1.0 - a12;
    // Formula:
    x_ = (x * a11 - y * a01) / (a00 * a11 - a01 * a10);
    y_ = (y * a00 - x * a10) / (a00 * a11 - a01 * a10);
    return cv::Point(x_, y_);
}
*/

//Compare contour in temp-coor 
/*
// Define compareContour function:
std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
compareContour(const std::vector<cv::Point>& temp_contour, std::map<std::pair<int, int>, std::vector<int>> temp_bin, const cv::Mat& targ_img, std::vector<cv::Point2d> rect, double threshold, int distance_thresh) {
    auto start = std::chrono::high_resolution_clock::now();

    // Sinh ra ma trận chuyển từ các cặp điểm trong rect: convert_mat biến đổi toạ độ từ targ_img lớn sang targ_img nhỏ, inv_convert_mat ngược lại
    int height = targ_img.rows;
    int width = targ_img.cols;
    std::vector<cv::Point2f> src = { cv::Point2f(0, 0), cv::Point2f(width - 1, 0), cv::Point2f(0, height - 1) };
    std::vector<cv::Point2f> dst = { cv::Point2f(rect[0]), cv::Point2f(rect[1]), cv::Point2f(rect[3]) };
    cv::Mat convert_mat = cv::getAffineTransform(src, dst);

    // Find matrix for inverse transformation
    cv::Mat convert_mat_full = cv::Mat::zeros(3, 3, convert_mat.type());
    convert_mat.copyTo(convert_mat_full(cv::Rect(0, 0, 3, 2)));
    convert_mat_full.at<double>(2, 0) = 0.0;
    convert_mat_full.at<double>(2, 1) = 0.0;
    convert_mat_full.at<double>(2, 2) = 1.0;
    cv::Mat inv_convert_mat_full = convert_mat_full.inv(cv::DECOMP_SVD);
    cv::Mat inv_convert_mat = inv_convert_mat_full(cv::Rect(0, 0, 3, 2));

    int patch_size = 2 * distance_thresh;

    // Find contour and divide points into bins
    std::vector<cv::Point> targ_contour = extract_contour(targ_img, threshold);

    std::vector<cv::Point> converted_targ_contour;
    std::vector<cv::Point2f> pos_1, pos_2;
    double min_distance;

    // Calculate distance 1 - from targ to temp
    //std::vector<double> distance_1(targ_contour.size());
    for (int i = 0; i < targ_contour.size(); i++) {
        const cv::Point& point = targ_contour[i];   // Chạy lần lượt các điểm thuộc targ_contour

        cv::Point targ_pnt = convert_coor(point, convert_mat);
        converted_targ_contour.push_back(cv::Point2f(targ_pnt));

        int x = targ_pnt.x / patch_size;
        int y = targ_pnt.y / patch_size;   // Lẩy chỉ số của bin chứa điểm đang xét

        // Lấy chỉ số của 3 bin lân cận cần xét
        int k1 = (targ_pnt.x % patch_size >= distance_thresh) ? 1 : -1;
        int k2 = (targ_pnt.y % patch_size >= distance_thresh) ? 1 : -1;
        std::vector<std::pair<int, int>> key_vec{ {x, y}, {x + k1, y}, {x, y + k2}, {x + k1, y + k2} };

        // Tạo list gồm các index trong các bin
        std::vector<int> idx_list;
        for (const auto& k : key_vec) {
            if (temp_bin.count(k) > 0) {
                idx_list.insert(idx_list.end(), temp_bin[k].begin(), temp_bin[k].end());
            }
        }
        if (idx_list.size() != 0) {     // Nếu list không rỗng, ta tính toán với các điểm trong list
            min_distance = std::numeric_limits<double>::max();
            for (const auto& idx : idx_list) {
                double distance = std::abs(temp_contour[idx].x - targ_pnt.x) + std::abs(temp_contour[idx].y - targ_pnt.y);
                min_distance = std::min(min_distance, distance);
            }
            //distance_1[i] = min_distance;
        }
        else {        // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
            //distance_1[i] = distance_thresh + 1;
            min_distance = distance_thresh + 1;
        }

        if (min_distance > distance_thresh) {
            pos_1.push_back(cv::Point2f(targ_contour[i]));
        }
    }

    // Calculate distance 2
    //std::vector<double> distance_2(temp_contour.size());
    std::map<std::pair<int, int>, std::vector<int>> targ_bin = devide_bin(converted_targ_contour, patch_size);
    for (int i = 0; i < temp_contour.size(); i++) {
        const cv::Point& point = temp_contour[i];
        int x = point.x / patch_size;
        int y = point.y / patch_size;
        int k1 = (point.x % patch_size >= distance_thresh) ? 1 : -1;
        int k2 = (point.y % patch_size >= distance_thresh) ? 1 : -1;
        std::vector<std::pair<int, int>> key_vec{ {x, y}, {x + k1, y}, {x, y + k2}, {x + k1, y + k2} };
        std::vector<int> idx_list;
        for (const auto& k : key_vec) {
            if (targ_bin.count(k) > 0) {
                idx_list.insert(idx_list.end(), targ_bin[k].begin(), targ_bin[k].end());
            }
        }

        if (idx_list.size() != 0) {
            min_distance = std::numeric_limits<double>::max();
            int min_index = 0;
            for (const auto& idx : idx_list) {
                double distance = std::abs(converted_targ_contour[idx].x - point.x) + std::abs(converted_targ_contour[idx].y - point.y);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = idx;
                }
            }
            //distance_2[i] = min_distance;
            if (min_distance > distance_thresh) {
                pos_2.push_back(cv::Point2f(targ_contour[min_index]));
            }
        }
        else {
            //distance_2[i] = distance_thresh + 1;
            min_distance = distance_thresh + 1;
            pos_2.push_back(cv::Point2f(convert_coor(temp_contour[i], inv_convert_mat)));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << std::endl;

    return { pos_1, pos_2 };

}*/