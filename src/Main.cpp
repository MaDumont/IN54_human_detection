#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

using namespace cv;
using namespace std;

void bodyParts(Mat img);
void drawRectangle(Point p1, Point p2, Mat img){

    vector<cv::Point> points;
    points.push_back(p1);
    points.push_back(p2);

    Rect brect = cv::boundingRect(cv::Mat(points).reshape(2));
    rectangle(img, brect.tl(), brect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);

}

double distanceBetweenTwoPoints(double x, double y, double a, double b) {
	return sqrt(pow(x - a, 2) + pow(y - b, 2));
}

int main(int argc, char** argv )
{
	if ( argc != 2 )
	{
        	printf("usage: DisplayImage.out <Image_Path>\n");
      	  	return -1;
   	}

	Mat img,imgGray, des_img;
	img = imread( argv[1], 1 );

	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	img = imread( argv[1], 1 ); //il suffit de donner le path du fichier dans l'entree du programme
	resize(img, des_img, Size(500, 900), 0, 0, INTER_LINEAR);
	

	vector<Rect> found, found_filtered;
	hog.detectMultiScale(des_img, found, 0, Size(4, 4), Size(8, 8), 1.05, 2);
		
	size_t i, j;
	double maxDistance = 0;
	Rect maxRect;
		
	for (i = 0; i<found.size(); i++)
	{
			Rect r = found[i];
			//rectangle(des_img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);// mettre les personnes dans des cadres
			double distance = distanceBetweenTwoPoints(r.tl().x, r.tl().y, r.br().x, r.br().y);
			if (distance > maxDistance)// chercher le rectangle le plus grand si il y a plusieurs personnes sur l image
			{
				maxRect = r;
				distance = maxDistance;
			}
		
	}

	bodyParts(des_img(maxRect));
		
	//imshow("image source", des_img);
	//imshow("corp image", des_img(maxRect));

	waitKey();
	return 0;
}

cv::Mat cropBinary(Mat binaryMat){
	Mat horizontal(binaryMat.cols,1,CV_32S);//horizontal histogram
	horizontal = Scalar::all(0);
	Mat vertical(binaryMat.rows,1,CV_32S);//vertical histogram	
	vertical = Scalar::all(0);
	
	// Count the number of white (non zero) pixels	
	for(int i=0;i<binaryMat.cols;i++)
	{
		horizontal.at<int>(i,0)=countNonZero(binaryMat(Rect(i,0,1,binaryMat.rows)));
	}
				 
	for(int i=0;i<binaryMat.rows;i++)
	{
		vertical.at<int>(i,0) = countNonZero(binaryMat(Rect(0,i,binaryMat.cols,1)));
	}
	
	//crop the image
	int top, bottom, left, right;
	//Top
	if(vertical.at<int>(0,0) == 0){
		for (int i = 1 ; i < binaryMat.rows ; ++i){
			if(vertical.at<int>(i-1,0) == 0 && vertical.at<int>(i,0) > 0){
				top = i;
				break;
			}
		}
	}
	else top = 0;

	//Bottom
	if(vertical.at<int>(binaryMat.rows-1,0) == 0){
		for (int i = binaryMat.rows-2 ; i > 1 ; --i){
			if(vertical.at<int>(i+1,0) == 0 && vertical.at<int>(i,0) > 0){
				bottom = i;
				break;
			}
		}	
	}
	else bottom = binaryMat.rows - 1;
	cout << bottom << endl;

	//Left
	if(horizontal.at<int>(0,0) == 0){
		for (int i = 1 ; i < binaryMat.cols ; ++i){
			if(horizontal.at<int>(i-1,0) == 0 && horizontal.at<int>(i,0) > 0){
				left = i;
				break;
			}
		}
	}
	else left = 0;

	//Right
	if(horizontal.at<int>(binaryMat.cols-1,0) == 0){
		for (int i = binaryMat.cols-2 ; i > 1 ; --i){
			if(horizontal.at<int>(i+1,0) == 0 && horizontal.at<int>(i,0) > 0){
				right = i;
				break;
			}
		}
	}
	else right = binaryMat.cols - 1;

	cout << top << " " << left << " " << bottom << " " << right << endl;
	//Crop the image and return it
	int width = right - left;
	int height = bottom - top;
	cout << width << " " << height << endl;
	cv::Rect bodyRect(left, top, width, height);
	cv::Mat croppedMat = binaryMat(bodyRect);
	return croppedMat;
}

void bodyParts (Mat img){
	//Points : Head, Head Opposite, Left foot, Right foot, Left Hand, Right hand
	std::unordered_map<std::string, cv::Point> bodyPoints;

	//Grayscale matrix
	cv::Mat grayscaleMat (img.size(), CV_8U);
	//Convert BGR to Gray
	cv::cvtColor( img, grayscaleMat, CV_BGR2GRAY );
	//Binary image
	cv::Mat binaryMat(grayscaleMat.size(), grayscaleMat.type());
	//Apply thresholding
	cv::threshold(grayscaleMat, binaryMat, 100, 255, cv::THRESH_BINARY);
	//Reverse the colors
	bitwise_not (binaryMat, binaryMat);

	binaryMat = cropBinary(binaryMat);
	
	Mat horizontal(binaryMat.cols,1,CV_32S);//horizontal histogram
	horizontal = Scalar::all(0);
	Mat vertical(binaryMat.rows,1,CV_32S);//vertical histogram	
	vertical = Scalar::all(0);
	
	// Count the number of white (non zero) pixels	
	for(int i=0;i<binaryMat.cols;i++)
	{
		horizontal.at<int>(i,0)=countNonZero(binaryMat(Rect(i,0,1,binaryMat.rows)));
	}
				 
	for(int i=0;i<binaryMat.rows;i++)
	{
		vertical.at<int>(i,0) = countNonZero(binaryMat(Rect(0,i,binaryMat.cols,1)));
	}

	//Find Head and Hips
	
	//Find Hands
	
	//Find Feet
	
	/*
	 * Ideas : Find maximum, cut image, spine starts from the head and go straight down TODO 
	 */

	imshow("image bin", binaryMat);

}

