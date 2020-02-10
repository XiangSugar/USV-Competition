#include "../sensors_include/vision.h"

#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <vector>

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
	Center_ = cv::Point2f(0.0, 0.0);
	Center1_ = cv::Point2f(0.0, 0.0);
	Center2_ = cv::Point2f(0.0, 0.0);
}

void colorDetecter::get_color_range(char targetColor)
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
		mins_ = 135;
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

	cout << "B——Black\n";
	//cout << "H——Gray\n";
	cout << "W——White\n";
	cout << "H——Red\n";
	cout << "O——Orange\n";
	cout << "Y——Yellow\n";
	cout << "G——Green\n";
	cout << "L——Blue\n";
	cout << "P——Purple\n";
	cout << "Please enter the letter corresponding to the target color: ";
}

void colorDetecter::set_FOV(double horizontal_fov)
{
	if (horizontal_fov < 0)
		std::cout << "ERROR：The horizontal's fov must be positive！\n";
	else
		horizontal_fov_ = horizontal_fov;
}

char  colorDetecter::get_detectedColor() const
{
	return detectedColor_;
}

double colorDetecter::get_angle(int flag)
{
	double hw = img_size_.width / 2.0;
	if (0 == flag)
		// The counterclockwise direction is positive
		angle_ = (hw - Center_.x) / img_size_.width * horizontal_fov_;
	else if (1 == flag)
		angle_ = (hw - Center1_.x) / img_size_.width * horizontal_fov_;
	else if (2 == flag)
		angle_ = (hw - Center2_.x) / img_size_.width * horizontal_fov_;
	else
	{
		std::cout << "Error: the value of param flag should be 0 or 1 or 2!";
		return 0;
	}
	return angle_;
}

void colorDetecter::get_color_mask(char targetColor, cv::Mat & fhsv, cv::Mat & mask)
{
	if ('H' == targetColor)
	{
		cv::Mat mask1;
		cv::Mat mask2;
		get_color_range('R');
		cv::inRange(fhsv, cv::Scalar(minh_, mins_, minv_), cv::Scalar(maxh_, maxs_, maxv_), mask1);
		get_color_range('r');
		cv::inRange(fhsv, cv::Scalar(minh_, mins_, minv_), cv::Scalar(maxh_, maxs_, maxv_), mask2);
		mask = mask1 + mask2;
	}
	else
	{
		get_color_range(targetColor);
		cv::inRange(fhsv, cv::Scalar(minh_, mins_, minv_), cv::Scalar(maxh_, maxs_, maxv_), mask);
	}
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1));
	cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
}

void colorDetecter::draw_result(cv::Mat & result, std::vector<std::vector<cv::Point>> & contours,
	int index, cv::Point center, float & radius)
{
	cv::drawContours(result, contours, index, cv::Scalar(0, 255, 0), 2, 8);
	cv::circle(result, center, (int)radius, cv::Scalar(0, 0, 255), 1);
}

void colorDetecter::draw_result(cv::Mat & result, std::vector<std::vector<cv::Point>> & contours,
	float & radius1, float & radius2)
{
	cv::drawContours(result, contours, 0, cv::Scalar(0, 255, 0), 2, 8);
	cv::circle(result, (cv::Point)Center1_, (int)radius1, cv::Scalar(0, 0, 255), 1);
	cv::drawContours(result, contours, 1, cv::Scalar(0, 255, 0), 2, 8);
	cv::circle(result, (cv::Point)Center2_, (int)radius2, cv::Scalar(0, 0, 255), 1);
}

void colorDetecter::find_longest_contour(cv::Mat & mask, std::vector<std::vector<cv::Point>> & contours,
	double & maxLen, int & index)
{
	double len;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	int size = contours.size();
	for (int t = 0; t < size; t++)
		if (maxLen < (len = cv::arcLength(contours[t], true)))
		{
			maxLen = len;
			index = t;
		}
}

int colorDetecter::find_maxLen(double * maxlen)
{
	int index = 0;
	for (int i = 0; i < 3; i++)
		if (maxlen[i] > maxlen[index])
			index = i;
	return index;
}

int colorDetecter::process(char targetColor, cv::Mat & result, runMode runmode, double minLen)
{
	using namespace cv;
	int state = 0;

	if (!targetColor)
	{
		std::cout << "Please enter the right letter corresponding to the target color.\n";
		return -1;
	}
	if (minLen < 0)
	{
		std::cout << "ERROR: The param @minLen must be positive.\n";
		return -1;
	}
	else
		minLen_ = minLen;

	cv::Mat img_Blur;
	cv::GaussianBlur(image_, img_Blur, cv::Size(3, 3), 0, 0);
	cv::Mat fhsv;
	cv::cvtColor(img_Blur, fhsv, COLOR_BGR2HSV);
	cv::Mat mask;
	get_color_mask(targetColor, fhsv, mask);
	//imshow("mask", mask);
	
	std::vector<std::vector<cv::Point>> contours;
	if ('G' == targetColor)
	{
		double len = 0;
		cv::findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		//calculate the length of each contour
		int size = contours.size();
		std::vector<double>  contour_size;
		for (int i = 0; i < size; i++)
		{
			len = cv::arcLength(contours[i], true);
			contour_size.push_back(len);
		}
		float radius1;
		float radius2;

		if (1 == size && (contour_size[0] > minLen_))
		{
			cv::minEnclosingCircle(contours[0], Center1_, radius1);
			state = 1;
			detectedColor_ = targetColor;
		}
		else if (size > 1)
		{
			// sort by contour's size in descending order
			cv::vector<cv::Point> temp_Contour;
			double temp_contour_size;
			for (int i = 0; i < size; i++)
			{
				for (int j = size - 1; j > i; j--)
				{
					if (contour_size[j] > contour_size[j - 1])
					{
						//for contour_size
						temp_contour_size = contour_size[j];
						contour_size[j] = contour_size[j - 1];
						contour_size[j - 1] = temp_contour_size;
						//for contours
						temp_Contour = contours[j];
						contours[j] = contours[j - 1];
						contours[j - 1] = temp_Contour;
					}
				}
			}

			if (contour_size[0] > minLen_)
			{
				cv::minEnclosingCircle(contours[0], Center1_, radius1);
				if (contour_size[1] > minLen_)
				{
					cv::minEnclosingCircle(contours[1], Center2_, radius2);
					state = 2;
				}
				else
					state = 1;
				detectedColor_ = targetColor;
			}
			else
				state = 0;
		}
		else
			state = 0;

		if (debug == runmode)
		{
			//cv::Mat result = cv::Mat::zeros(img_size_, CV_8UC3);
			if (2 == state)
				draw_result(result, contours, radius1, radius2);
			else if (1 == state)
				draw_result(result, contours, 0, Center1_, radius1);
			else
				;
			//cv::imshow("result", result);
			/* for debug
			cv::imshow("mask", mask);
			*/
		}
		return state;

	}
	else
	{
		int index;
		double maxLen = 0;
		find_longest_contour(mask, contours, maxLen, index);
		if (maxLen < minLen_)
		{
			detectedColor_ = 'N';	// find no target color
			return -1;
		}
		else
		{
			float radius;
			cv::minEnclosingCircle(contours, Center_, radius);
			detectedColor_ = targetColor;

			if (debug == runmode)
			{
				draw_result(result, contours, index, Center_, radius);
				/* for debug
				cv::imshow("mask", mask);
				*/
			}
			return 0;
		}
	}
}


// 可以将三种颜色的检测轮廓都画出来  计算三个角度
int colorDetecter::process(cv::Mat & result, runMode runmode, double minLen)
{
	using namespace cv;
	if (minLen < 0)
	{
		std::cout << "ERROR: The param @minLen must be positive.\n";
		return false;
	}
	else
		minLen_ = minLen;

	double len = 0;
	cv::Mat fhsv;
	int index = 0;
	double maxLen_H = 0;
	double maxLen_B = 0;
	double maxLen_L = 0;
	int index_H = 0;
	int index_B = 0;
	int index_L = 0;
	cv::Mat mask_H;
	cv::Mat mask_B;
	cv::Mat mask_L;
	std::vector<std::vector<cv::Point>> contours_H;
	std::vector<std::vector<cv::Point>> contours_B;
	std::vector<std::vector<cv::Point>> contours_L;
	cvtColor(image_, fhsv, COLOR_BGR2HSV);

	get_color_mask('H', fhsv, mask_H);
	find_longest_contour(mask_H, contours_H, maxLen_H, index_H);
	get_color_mask('B', fhsv, mask_B);
	find_longest_contour(mask_B, contours_B, maxLen_B, index_B);
	get_color_mask('L', fhsv, mask_L);
	find_longest_contour(mask_L, contours_L, maxLen_L, index_L);

	double maxlen[3] = { maxLen_H, maxLen_B, maxLen_L };
	/* for debug
	int j = 0;
	while (j < 3) {
		std::cout << maxlen[j] << std::endl;
		j++;
	}
	*/
	index = find_maxLen(maxlen);
	if (maxlen[index] < minLen_)
	{
		detectedColor_ = 'N';	// find no target color
		return 0;
	}
	else
	{
		float radius;
		if (0 == index)
		{
			cv::minEnclosingCircle(contours_H[index_H], Center_, radius);
			detectedColor_ = 'H';
		}
		else if (1 == index)
		{
			cv::minEnclosingCircle(contours_B[index_B], Center_, radius);
			detectedColor_ = 'B';
		}
		else if (2 == index)
		{
			cv::minEnclosingCircle(contours_L[index_L], Center_, radius);
			detectedColor_ = 'L';
		}
		else
			std::cout << "Error index!" << std::endl;
		if (debug == runmode)
		{
			if (0 == index)
				draw_result(result, contours_H, index_H, Center_, radius);
			else if (1 == index)
				draw_result(result, contours_B, index_B, Center_, radius);
			else if (2 == index)
				draw_result(result, contours_L, index_L, Center_, radius);
			else
				std::cout << "Error index!" << std::endl;

			/* for debug
			cv::imshow("mask_H", mask_H);
			cv::imshow("mask_B", mask_B);
			cv::imshow("mask_L", mask_L);
			*/
		}
		return 1;
	}
}

