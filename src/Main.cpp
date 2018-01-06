#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>

#include <opencv2/core/utility.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "main.h"

using namespace cv;
using namespace std;

//Global var
Mat binaryCorpImage;
int scale;
float bodyLength;


class WatershedSegmenter{
private:
     Mat markers;
public:
    void setMarkers( Mat& markerImage)
    {
        markerImage.convertTo(markers, CV_32S);
    }

     Mat process( Mat &image)
    {
         watershed(image, markers);
        markers.convertTo(markers,CV_8U);
        return markers;
    }
};

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

int main(int argc, char** argv){
	if ( argc != 2 )
	{
        	printf("usage: DisplayImage.out <Image_Path>\n");
      	  	return -1;
   	}

	// Get image from path
	Mat img, des_img, bodyEdge;
	Rect bodyRect;
	bodyLength = 180.0f;
	img = imread( argv[1], 1 );
	//resize(img, des_img, Size(500, 1000), 0, 0, INTER_LINEAR);
	 resize(img, des_img, Size(), 0.50, 0.50);

	//des_img = img;
	// Processing
	bodyRect = bodyDetect(des_img);
	
	bodyEdge = edgeDetect(des_img(bodyRect));

	//bodyParts(des_img); //USE THIS IF THE INPUT IS A BINARY IMAGE
	//bodyParts(des_img(bodyRect));

	//imshow("image source", des_img);
	//imshow("corp image", des_img(maxRect));

//	measureBodyParts(des_img, bodyParts(des_img));


	waitKey();
}

/*
 * Input : two points
 * Output : median of the two points
 */
Point median(Point a, Point b) {
	Point ret;
	ret.x = (a.x + b.x) / 2;
	ret.y = (a.y + b.y) / 2;
	return ret;
}

/*
 * Draw the distances between two points
 */
void draw(Mat img,double distance,Point p1,Point p2) 
{	
	std::ostringstream output;
	std::string distance_in_string;
	distance_in_string = (dynamic_cast< std::ostringstream*>(&(output << distance <<std::endl)))->str();
	cv::line(img, p1, p2, (20, 60, 50), 5);
	cv::putText(img, distance_in_string, median(p1, p2), cv::FONT_HERSHEY_SIMPLEX, 0.5, (20, 100, 100), 2, CV_AA);

}

/*
 * get scale number of pixel for 1 cm
 */
int getImageScale(Mat img,Point p1,Point p2)
{
	double d = distanceBetweenTwoPoints(p1.x, p1.y, p2.x, p2.y);
	return (int)(d / bodyLength);
}

/*
 * mesure body shapes
 */
void measureBodyParts(Mat img,pointMap bodyShapes) 
{	
	std::unordered_map<std::string, Point>::iterator head;
	std::unordered_map<std::string, Point>::iterator shoulderL;
	std::unordered_map<std::string, Point>::iterator shoulderR;
	std::unordered_map<std::string, Point>::iterator handL;
	std::unordered_map<std::string, Point>::iterator handR;
	std::unordered_map<std::string, Point>::iterator footR;
	std::unordered_map<std::string, Point>::iterator footL;
	std::unordered_map<std::string, Point>::iterator hipL;
	std::unordered_map<std::string, Point>::iterator hipR;

	head = bodyShapes.find("Head");
	handR = bodyShapes.find("RightHand");
	handL = bodyShapes.find("LeftHand");
	shoulderL = bodyShapes.find("LeftShoulder");
	shoulderR = bodyShapes.find("RightShoulder");
	footL = bodyShapes.find("LeftFoot");
	footR = bodyShapes.find("RightFoot");
	hipL = bodyShapes.find("LeftHip");
	hipR = bodyShapes.find("RightHip");

	double lengthLeftArm = distanceBetweenTwoPoints(handL->second.x, handL->second.y, shoulderL->second.x, shoulderL->second.y)/scale;
	double lengthRightArm = distanceBetweenTwoPoints(handR->second.x, handR->second.y, shoulderR->second.x, shoulderR->second.y)/scale;
	double lengthLeftFoot = distanceBetweenTwoPoints(footL->second.x, footL->second.y, hipL->second.x, hipL->second.y) / scale;
	double lengthRightFoot = distanceBetweenTwoPoints(footR->second.x, footR->second.y, hipR->second.x, hipR->second.y) / scale;

	draw(binaryCorpImage, lengthLeftArm, handL->second, shoulderL->second);
	draw(binaryCorpImage, lengthRightArm, handR->second, shoulderR->second);
	draw(binaryCorpImage, lengthLeftFoot, footR->second, hipR->second);
	draw(binaryCorpImage, lengthLeftFoot, footL->second, hipL->second);
	
	int block = img.rows / 8;
	float anthropometryArm = block * 3.5;
	float anthropometryfoot = block * 5;
	
	cout << "length Left Arm : " << lengthLeftArm << " cm ; taux 'd'erreur = " << abs((anthropometryArm - lengthLeftArm)/ anthropometryArm )/100<< endl;
	cout << "length Right Arm : " << lengthRightArm << " cm; taux 'd'erreur = " << abs((anthropometryArm - lengthRightArm) / anthropometryArm) / 100 << endl;
	cout << "length Left Foot : " << lengthLeftFoot << " cm; taux 'd'erreur = " << abs((anthropometryfoot - lengthLeftFoot) / anthropometryfoot) / 100 << endl;
	cout << "length Right Foot : " << lengthRightFoot << " cm; taux 'd'erreur = " << abs((anthropometryfoot - lengthRightFoot) / anthropometryfoot) / 100 << endl;

	imshow("binary corp image after mesure", binaryCorpImage);

}

/*
 * Extract a rectangle containing the body
 */
cv::Rect bodyDetect(cv::Mat image)
{
	Mat img;
	img = image;

	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	vector<Rect> found, found_filtered;
	hog.detectMultiScale(img, found, 0, Size(4, 4), Size(8, 8), 1.05, 2);
		
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
		
	return maxRect;
}


/*
	extract the edge of the human body
*/
cv::Mat edgeDetect(cv::Mat rectContourHumain){


	Mat dest, img_gray, img_bin;



	int cols = rectContourHumain.cols;
	int rows = rectContourHumain.rows;




	// Create markers image
	Mat markers(rectContourHumain.size(),CV_8U, Scalar(-1));
	//Rect(topleftcornerX, topleftcornerY, width, height);

	//HOLD LEFT TO HEAD
//	markers(Rect(5,5,cols/3,rows/5)) = Scalar::all(1);

	//HOLD RIGHT TO HEAD
//	markers(Rect(2*cols/3,5,cols/3-5,rows/5)) = Scalar::all(1);

	//left rectangle
	markers(Rect(0,0,cols, 5)) = Scalar::all(1);
	//bottom rectangle
	markers(Rect(0,rows-5,cols, 5)) = Scalar::all(1);
	//left rectangle
	markers(Rect(0,0,5,rows)) = Scalar::all(1);
	//right rectangle
	markers(Rect(cols-5,0,5,rows)) = Scalar::all(1);
	//centre rectangle

	int centreW = cols/4;
	int centreH = rows/4;
	markers(Rect((cols/2)-(centreW/2),(rows/2)-(centreH/2), centreW, centreH)) = Scalar::all(2);
	markers.convertTo(markers,CV_BGR2GRAY);

	Point p1 = Point((cols/2)-(centreW/2),(rows/2)-(centreH/2)); 
	Point p2 = Point((cols/2)+(centreW/2),(rows/2)+(centreH/2));
	//drawRectangle(p1,p2,rectContourHumain);



	//Create watershed segmentation object
	WatershedSegmenter segmenter;
	segmenter.setMarkers(markers);
	Mat wshedMask = segmenter.process(rectContourHumain);
	Mat mask;
	convertScaleAbs(wshedMask, mask, 1, 0);
	double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);
	bitwise_and(rectContourHumain, rectContourHumain, dest, mask);
	dest.convertTo(dest,CV_8U);

	/// Convert the image to grayscale
	cvtColor( dest, img_gray, COLOR_BGR2GRAY );
	threshold(img_gray, img_bin, 1, 255, CV_THRESH_BINARY_INV);

	imshow("bin", img_bin);
	return img_bin;
}
/*
 * Output : The horizontal histogram of a binary image
 */
cv::Mat horizontalProj(Mat binaryMat){
	Mat horizontal(binaryMat.cols, 1, CV_32S);
	horizontal = Scalar::all(0);

	// Count the number of white (non zero) pixels	
	for(int i=0;i<binaryMat.cols;i++)
	{
		horizontal.at<int>(i,0)=countNonZero(binaryMat(Rect(i,0,1,binaryMat.rows)));
	}

	return horizontal;
}

/*
 * Output : the vertical histogram of a binary image
 */
cv::Mat verticalProj(Mat binaryMat){
	Mat vertical(binaryMat.rows, 1, CV_32S);
	vertical = Scalar::all(0);

	for(int i=0;i<binaryMat.rows;i++)
	{
		vertical.at<int>(i,0) = countNonZero(binaryMat(Rect(0,i,binaryMat.cols,1)));
	}	

	return vertical;
}

/*
 * Compute an opening on the binaryMat
 */
cv::Mat opening(Mat binaryMat){
	Mat se1 = getStructuringElement(MORPH_RECT, Size(2, 2));
	
	Mat mask;
	morphologyEx(binaryMat, mask, MORPH_OPEN, se1);
	
	Mat output(binaryMat.rows, binaryMat.cols, CV_32S);
	output = Scalar::all(0);
	output = binaryMat.clone();
	output.setTo(Scalar(0), mask == 0);
	
	return output;
}

/*
 * Compute a closing on the binaryMat
 */
cv::Mat closing(Mat binaryMat){
	//Define the structuring element
	Mat se1 = getStructuringElement(MORPH_RECT, Size(5, 5));

	Mat mask;
	morphologyEx(binaryMat, mask, MORPH_CLOSE, se1);
	
	Mat output(binaryMat.rows, binaryMat.cols, CV_32S);
	output = Scalar::all(0);
	output = binaryMat.clone();
	output.setTo(Scalar(0), mask == 0);
	
	return output;
}

/*
 * Input : a binary image of the body
 * Output : a cropped binary image (removal of the unnecessary lines and columns containing only black pixels)
 */
cv::Rect cropBinary(Mat binaryMat){
	Mat horizontal = horizontalProj(binaryMat);
	Mat vertical = verticalProj(binaryMat);
	
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

	//Crop the image
	int width = right - left + 1;
	int height = bottom - top ;
	cv::Rect bodyRect(left, top, width, height);
	//cv::Mat croppedMat = binaryMat(bodyRect);
	return bodyRect;
}

/*
 * Input : a cropped binary image
 * Output : a hashmap containing all the points of the body parts
 */
pointMap bodyParts (Mat img){
	//Points : Head, Left foot, Right foot, Left Hand, Right hand, ...
	pointMap bodyPoints;
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

	//Improve the image quality
	binaryMat = closing(binaryMat);
	binaryMat = opening(binaryMat);
	cv::Rect cropRect = cropBinary(binaryMat);
	binaryMat = binaryMat(cropRect);
	cv::Mat original = img(cropRect);

	bodyPoints.emplace("Head", findHead(binaryMat));
	bodyPoints.emplace("RightHand", findHand(binaryMat, false));
	bodyPoints.emplace("LeftHand", findHand(binaryMat, true));
	bodyPoints.emplace("RightFoot", findFoot(binaryMat, false));
	bodyPoints.emplace("LeftFoot", findFoot(binaryMat, true));
	bodyPoints.emplace("RightShoulder", findShoulder(binaryMat, false));
	bodyPoints.emplace("LeftShoulder", findShoulder(binaryMat, true));
	int bodyCenter = (int)bodyPoints.find("Head")->second.x; 
	bodyPoints.emplace("RightHip", findHip(binaryMat, bodyCenter, false));
	bodyPoints.emplace("LeftHip", findHip(binaryMat, bodyCenter, true));

	//get the scale en cm of the image //khaled
	scale = getImageScale(binaryMat, bodyPoints.find("Head")->second, median(bodyPoints.find("LeftFoot")->second, bodyPoints.find("RightFoot")->second));
	cout << " scale in cm = " << scale << endl;
	//***********************************************************************	

	//Display the coordinates of the points and draw a circle around them
	for (auto& x: bodyPoints) {
		std::cout << x.first << ": " << x.second << std::endl;
		circle(binaryMat, x.second, 5, Scalar(0,0,255), 4);
	}
	imshow("image bin", original);

	cv::cvtColor(binaryMat, binaryCorpImage, CV_GRAY2RGB);
	
	return bodyPoints;
}

//Find the top of the head : we look for the white pixels on the first line. We assume the median pixel is the top of the head.
cv::Point findHead(Mat bodyImg){
	cv::Point head;
	cv::Mat line, nonZero;

	line = bodyImg(cv::Rect(0,0,bodyImg.cols,1));
	cv::findNonZero(line, nonZero);

	int median = (int)nonZero.total() / 2;
	head = nonZero.at<Point>(median);

	return head;
}

/*
 * Find the extremity of the hands
 * side : 0 for left hand
 * 	  1 for right hand
 * Output : the point corresponding to the extremity of a hand
 */
cv::Point findHand(Mat bodyImg, bool side){
	cv::Point hand;
	cv::Mat searchSpace, nonZero;
	int colNumber;

	if(side == false) colNumber = 0; // We look for the left hand (the right hand actually, but let's call it left hand for the sake of clarity)
	else if (side == true) colNumber = bodyImg.cols-1; // We look for the right hand

	searchSpace = bodyImg(cv::Rect(colNumber,0,1,bodyImg.rows));
	cv::findNonZero(searchSpace, nonZero);
	
	int median = (int)nonZero.total() / 2;
	hand = nonZero.at<Point>(median);
	
	//Conversion to match the coordinates of the original image
	if(side == true) hand.x = bodyImg.cols-1;

	return hand;
}

/*
 * Find the extremity of the feet
 * Assumption : there should be enough space between the feet
 * side : 0 for left foot
 * 	  1 for right foot
 * Output : the point corresponding to the extremity of a foot
 */
cv::Point findFoot(Mat bodyImg, bool side){
	cv::Point foot;
	cv::Mat searchSpace, correctedSearchSpace, nonZero;
	int colNumber;

	if(side == false) colNumber = 0;
	else if(side == true) colNumber = (int)(bodyImg.cols-1)/2; //We cut the image vertically

	searchSpace = bodyImg(cv::Rect(colNumber, bodyImg.rows-1, (bodyImg.cols-1)/2, 1));
	cv::findNonZero(searchSpace, nonZero);
	
	int median = (int)nonZero.total() / 2;
	foot = nonZero.at<Point>(median);

	//Conversion to match the coordinates of the original image
	foot.y = bodyImg.rows-1;
	if(side == true) foot.x = foot.x + (int)(bodyImg.cols-1)/2;

	return foot;
}

/*
 * Find the shoulders
 * side : 0 for left shoulder
 * 	  1 for right shoulder
 * Output : the point corresponding to a shoulder
 */
cv::Point findShoulder(Mat bodyImg, bool side){
	cv::Point shoulder;
	cv::Mat searchSpace, nonZero;

	cv::Mat horizontal = horizontalProj(bodyImg);
	cv::Mat vertical = verticalProj(bodyImg);


	int shouldersLine = 0;
	int spaceBetween = (int)bodyImg.rows / 30; //Estimation of the number of rows between the shoulders and the neck/head
	int startingRow = 2*spaceBetween;

	// Find the x coordinates of the shoulders
	for (int i = startingRow ; i < bodyImg.rows ; ++i){
		if(vertical.at<int>(i,0) > 2 * vertical.at<int>(i-spaceBetween,0)){
			shouldersLine = i;
			break;
		}
	}
	
	searchSpace = bodyImg(cv::Rect(0, shouldersLine, bodyImg.cols, 1));

	// Find the y coordinate of the requested shoulder
	cv::findNonZero(searchSpace, nonZero);
	if(side == false) shoulder = nonZero.at<Point>(0);
	if(side == true) shoulder = nonZero.at<Point>(nonZero.rows-1);
	
	shoulder.y = shouldersLine;	
	return shoulder;	
}

/*
 * Find the hip
 * side : 0 for left hip
 * 	  1 for right hip
 * Output : the point corresponding to a hip
 */
cv::Point findHip(Mat bodyImg, int bodyCenter, bool side){
	cv::Point hip;
	cv::Mat searchSpace, nonZero;
	int hipPos;

	searchSpace = bodyImg(cv::Rect(bodyCenter, 0, 1, bodyImg.rows));

	// Find the x coordinate of the hips
	cv::Mat vertical = verticalProj(searchSpace);
	for (int i = 2 ; i < bodyImg.rows ; ++i){
		if(vertical.at<int>(i,0) == 0 && vertical.at<int>(i-1,0) > 0){
			hipPos = i;
			break;
		}
	}

	// Find the y coordinate of the requested hip
	searchSpace = bodyImg(cv::Rect(0, hipPos, bodyImg.cols, 1));
	cv::findNonZero(searchSpace, nonZero);
	int div = (int)nonZero.total() / 4;
	if (side == false) hip = nonZero.at<Point>(div);
	if (side == true) hip = nonZero.at<Point>(3*div);
	hip.y = hipPos;	
	
	return hip;
}
