// use_vision.cpp -- example of using the colorDetecter class

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "../sensors_include/vision.h"

int main()
{
	return 0;
}

int image_proc() {
	using namespace cv;
	double angle;
	char det_color;

	Mat image = imread("E:\\Code library\\USV-Competition\\USV\\test_materials\\Red.jpg");
	imshow("image", image);
	colorDetecter color_det_img(image);

	if (color_det_img.process(colorDetecter::runMode::debug))
	{
		angle = color_det_img.get_angle();
		det_color = color_det_img.get_detectedColor();
		std::cout << "angle = " << angle << std::endl;
		std::cout << "detected color: " << det_color << std::endl;
	}
	else
		std::cout << "No targert color detected!\n";
	waitKey(0);
	return 0;
}