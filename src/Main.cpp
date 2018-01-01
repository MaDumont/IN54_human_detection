#include <stdio.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "Main.h"


using namespace cv;
using namespace std;

Mat src, src_gray;
Mat dst, detected_edges;
Mat img_bw;


/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
/** @function main */
int main( int argc, char** argv )
{
  /// Load an image
  src = imread( argv[1] );

  if( !src.data )
  { return -1; }

  /// Convert the image to grayscale
  cvtColor( src, src_gray, COLOR_BGR2GRAY );

  threshold(src_gray, img_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

  imshow("test",img_bw);

  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
  
}

