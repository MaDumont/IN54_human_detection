#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

using namespace cv;

void drawRectangle(Point p1, Point p2, Mat img);
double distanceBetweenTwoPoints(double x, double y, double a, double b);
cv::Rect bodyDetect(cv::Mat image);
cv::Mat cropBinary(Mat binaryMat);
void bodyParts (Mat img);

