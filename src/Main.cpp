#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

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
			rectangle(des_img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);// mettre les personne dans des cadres
			double distance = distanceBetweenTwoPoints(r.tl().x, r.tl().y, r.br().x, r.br().y);
			if (distance > maxDistance)// chercher la rectangle le plus grand si il ya plusieur personne sur l image
			{
				maxRect = r;
				distance = maxDistance;
			}
		
	}
		
	imshow("image source", des_img);
	imshow("corp image", des_img(maxRect));




	waitKey();
	return 0;

}
