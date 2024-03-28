#include "roi.h"
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

inline bool sortByX(const Point2d& a, const Point2d& b)
{
    return a.x < b.x;
}
inline bool sortByY(const Point2d& a, const Point2d& b)
{
    return a.y < b.y;
}

RectangleRoi::RectangleRoi(double x, double y, double width, double height, double angle) : Roi() {
    this->x = x;
    this->y = y;
    this->width = width;
    this->height = height;
    this->angle = angle;
}

RectangleRoi::RectangleRoi(Rect2d rect, double angle) : Roi() {
    this->x = rect.x;
    this->y = rect.y;
    this->width = rect.width;
    this->height = rect.height;
    this->angle = angle;
}

RectangleRoi::RectangleRoi(Rect rect, double angle) : Roi() {
    this->x = rect.x;
    this->y = rect.y;
    this->width = rect.width;
    this->height = rect.height;
    this->angle = angle;
}

RectangleRoi::RectangleRoi(Point2d point, Size2d size, double angle) : Roi() {
    this->x = point.x;
    this->y = point.y;
    this->width = size.width;
    this->height = size.height;
    this->angle = angle;
}

Rect RectangleRoi::boundingRect() {
    if (!isEmpty()) {
        if(angle == 0) {
            return Rect(xInt(), yInt(), widthInt(), heightInt());
        }
        else {
            vector<Point2d> vertices = getVertices();
            std::sort(std::begin(vertices), std::end(vertices), sortByX);
            double _x = vertices[0].x;
            double _width = vertices[3].x - vertices[0].x;

            std::sort(std::begin(vertices), std::end(vertices), sortByY);
            double _y = vertices[0].y;
            double _height = vertices[3].y - vertices[0].y;
            // Để đảm bảo vị trí top-right và bottom-right vẫn giữ nguyên độ chính xác cần chỉnh lại width với height
            // sau khi x,y bị làm tròn
            double _width2 = _x - (int)(_x+0.5) + _width;
            double _height2 = _x - (int)(_y+0.5) + _height;
            //delete vertices;
            return Rect((int)(_x+0.5), (int)(_y+0.5), (int)(_width2+0.5), (int)(_height2+0.5));
        }
    }
    else return Rect(0,0,0,0);
}

bool RectangleRoi::isEmpty() {
    return !((width > 0) && (height > 0));
}

vector<Point2d> RectangleRoi::getVertices() {
    if (!isEmpty())
    {
        vector<Point2d> vts  {
            Point2d(x, y),
            Point2d(x + width, y),
            Point2d(x + width, y + height),
            Point2d(x, y + height)
        };
        if (angle == 0) {
            return vts;
        }
        double angle_radian = -angle * CV_PI / 180;
        Point2d center = Point2d(x + width / 2, y + height / 2);
        //static Point2d pts[4];
        for (int i = 0; i < 4; i++)
        {
            Point2d currPoint = vts[i];
            // convert coordinator
            Point2d p = Point2d(currPoint.x - center.x, center.y - currPoint.y);
            // rotate point
            double x = p.x * cos(angle_radian) - p.y * sin(angle_radian);
            double y = p.x * sin(angle_radian) + p.y * cos(angle_radian);
            // convert to original coordinator
            vts[i] = Point2d(center.x + x, center.y - y);
        }
        return vts;
    }
    else return vector<Point2d>();
}

int RectangleRoi::xInt() {return (int)(x+0.5); }
int RectangleRoi::yInt() {return (int)(y+0.5); }
int RectangleRoi::widthInt() {return (int)(width+0.5); }
int RectangleRoi::heightInt() {return (int)(height+0.5); }