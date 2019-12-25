// use_vision.cpp -- exmple of using the colorDetecter class

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "../sensors_include/vision.h"

int main() {
	using namespace cv;
	double angle;
	char det_color;

	Mat image = imread("E:\\Code library\\USV-Competition\\USV\\test_materials\\Red.jpg");
	imshow("image", image);
	colorDetecter red_color_det(image);
	if (red_color_det.process('H', 'D'))
	{
		angle = red_color_det.get_angle();
		det_color = red_color_det.get_detectedColor();
		std::cout << "angle = " << angle << std::endl;
		std::cout << "detected color: " << det_color << std::endl;
	}
	else
		std::cout << "No targert color detected!\n";
	waitKey(0);
	return 0;
}