#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

using namespace cv;

/*********** TYPES ****************/
typedef std::unordered_map<std::string, cv::Point> pointMap;
typedef std::unordered_map<std::string, cv::Point>::const_iterator pointIterator;

/********* FUNCTIONS *************/
void drawRectangle(cv::Point p1, cv::Point p2, cv::Mat img);
double distanceBetweenTwoPoints(double x, double y, double a, double b);
cv::Rect bodyDetect(cv::Mat image);
cv::Mat cropBinary(cv::Mat binaryMat);
pointMap bodyParts (cv::Mat img);
cv::Point findHead (cv::Mat img);
cv::Point findLeftHand (cv::Mat img);
cv::Point findRightHand (cv::Mat img);
cv::Point findLeftFoot (cv::Mat img);
cv::Point findRightFoot (cv::Mat img);
cv::Point findWaist (cv::Mat img);
