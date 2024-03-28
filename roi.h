#ifndef ROI_H
#define ROI_H

#include <vector>
#include <opencv2/opencv.hpp>

class Roi
{

public:
    Roi() {}
    std::string ID = "";
    virtual cv::Rect boundingRect() = 0;
    virtual bool isEmpty() = 0;
    virtual std::vector<cv::Point2d> getVertices() = 0;
    virtual ~Roi() = default;
};

class RectangleRoi : public Roi {
public:
    RectangleRoi() : Roi() {}
    RectangleRoi(double x, double y, double width, double height, double angle = 0);
    RectangleRoi(cv::Point2d point, cv::Size2d size, double angle = 0);
    RectangleRoi(cv::Rect2d rect, double angle = 0);
    RectangleRoi(cv::Rect rect, double angle = 0);
    cv::Rect boundingRect() override;
    bool isEmpty() override;
    std::vector<cv::Point2d> getVertices() override;
    double x;
    double y;
    double width;
    double height;
    double angle;
    int xInt();
    int yInt();
    int widthInt();
    int heightInt();
};

#endif // ROI_H
