#include "../sensors_include/vision.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

colorDetecter::colorDetecter()       //defualt constructor
{

}

colorDetecter::colorDetecter(cv::Mat image, double minLen, double horizontal_fov)
{
	image_ = image;
	minLen_ = minLen;
	detectedColor_ = 'N';
	horizontal_fov_ = horizontal_fov;
	angle_ = 0.0;
	img_size_ = cv::Size(image.cols, image.rows);
	Center_ = cv::Point2d(0.0, 0.0);
}

void colorDetecter::dealColor(char targetColor)
{
	using std::cout;
	using std::endl;

	switch (targetColor) {
	case 'B':	//black
		minh_ = 0;
		maxh_ = 180;
		mins_ = 0;
		maxs_ = 255;
		minv_ = 0;
		maxv_ = 55;
		break;
	/*
	case 'H':	//grey
		minh_ = 0;
		maxh_ = 180;
		mins_ = 0;
		maxs_ = 43;
		minv_ = 46;
		maxv_ = 220;
		break;
	*/
	case 'W':	//white
		minh_ = 0;
		maxh_ = 180;
		mins_ = 0;
		maxs_ = 30;
		minv_ = 221;
		maxv_ = 255;
		break;
	case 'R':	//red
		minh_ = 0;
		maxh_ = 10;
		mins_ = 120;
		maxs_ = 255;
		minv_ = 50;
		maxv_ = 255;
		break;
	case 'r':	//red
		minh_ = 156;
		maxh_ = 179;
		mins_ = 100;
		maxs_ = 255;
		minv_ = 50;
		maxv_ = 255;
		break;
	case 'O':	//orange
		minh_ = 11;
		maxh_ = 25;
		mins_ = 43;
		maxs_ = 255;
		minv_ = 46;
		maxv_ = 255;
		break;
	case 'Y':	//yellow
		minh_ = 26;
		maxh_ = 34;
		mins_ = 43;
		maxs_ = 255;
		minv_ = 46;
		maxv_ = 255;
		break;
	case 'G':	//green
		minh_ = 35;
		maxh_ = 77;
		mins_ = 100;
		maxs_ = 255;
		minv_ = 46;
		maxv_ = 255;
		break;
	case 'L':	//blue
		minh_ = 100;
		maxh_ = 124;
		mins_ = 100;
		maxs_ = 255;
		minv_ = 46;
		maxv_ = 255;
		break;
	case 'P':	//purple
		minh_ = 125;
		maxh_ = 155;
		mins_ = 43;
		maxs_ = 255;
		minv_ = 46;
		maxv_ = 255;
		break;
	default:
		cout << "ERROR: This color can not be handled!" << endl;
		exit(0);
	}
}

void colorDetecter::helpText()
{
	using std::cout;

	cout << "B¡ª¡ªBlack\n";
	//cout << "H¡ª¡ªGray\n";
	cout << "W¡ª¡ªWhite\n";
	cout << "H¡ª¡ªRed\n";
	cout << "O¡ª¡ªOrange\n";
	cout << "Y¡ª¡ªYellow\n";
	cout << "G¡ª¡ªGreen\n";
	cout << "L¡ª¡ªBlue\n";
	cout << "P¡ª¡ªPurple\n";
	cout << "Please enter the letter corresponding to the target color:";
}

void colorDetecter::set_FOV(double horizontal_fov)
{

	if (horizontal_fov < 0)
		std::cout << "ERROR£ºThe horizontal's fov must be positive£¡\n";
	else
		horizontal_fov_ = horizontal_fov;
}

char  colorDetecter::get_detectedColor() const
{
	return detectedColor_;
}

double colorDetecter::get_angle()
{
	double hw = img_size_.width / 2.0;

	// The counterclockwise direction is positive
	angle_ = (hw - Center_.x) / img_size_.width * horizontal_fov_;

	return angle_;
}

bool colorDetecter::process(char targetColor, char runMode, double minLen)
{
	using namespace cv;
	if (minLen < 0)
	{
		std::cout << "Please enter the letter corresponding to the target color.\n";
		return false;
	}
	else
		minLen_ = minLen;

	double len = 0;
	double maxLen = 0;
	int index = 0;
	cv::Mat mask;

	cv::Mat img_Blur;
	cv::GaussianBlur(image_, img_Blur, cv::Size(3, 3), 0, 0);
	cv::Mat fhsv;
	cvtColor(img_Blur, fhsv, COLOR_BGR2HSV);

	if ('H' == targetColor)
	{
		cv::Mat mask1;
		cv::Mat mask2;
		dealColor('R');
		cv::inRange(fhsv, Scalar(minh_, mins_, minv_), Scalar(maxh_, maxs_, maxv_), mask1);
		dealColor('r');
		cv::inRange(fhsv, Scalar(minh_, mins_, minv_), Scalar(maxh_, maxs_, maxv_), mask2);
		mask = mask1 + mask2;
	}
	else
	{
		dealColor(targetColor);
		cv::inRange(fhsv, Scalar(minh_, mins_, minv_), Scalar(maxh_, maxs_, maxv_), mask);
	}
	//imshow("mask", mask);

	Mat kernel = cv::getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	cv::morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1, -1), 1);
	std::vector< std::vector<cv::Point>> contours;
	cv::findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	//cout << "The number of contours: " << contours.size() << endl;
	for (int t = 0; t < contours.size(); t++)
		if (maxLen <= (len = cv::arcLength(contours[t], true)))
		{
			maxLen = len;
			index = t;
		}

	if (maxLen < minLen_)
	{
		detectedColor_ = 'N';	// find no target color
		return false;
	}
	else
	{
		float radius;
		cv::minEnclosingCircle(contours[index], Center_, radius);
		if ('D' == runMode)
		{
			cv::Mat result = cv::Mat::zeros(img_size_, CV_8UC3);
			cv::drawContours(result, contours, index, Scalar(0, 255, 0), 2, 8);
			cv::circle(result, (cv::Point)Center_, (int)radius, Scalar(0, 0, 255), 1);
			cv::imshow("result", result);
		}
		detectedColor_ = targetColor;
		return true;
	}
}
