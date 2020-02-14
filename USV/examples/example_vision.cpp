// use_vision.cpp -- example of using the colorDetecter class

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "../sensors_include/vision.h"

int sgl_clr_image_proc();
int mu_clr_image_proc();
int sgl_clr_video_proc();
int mu_clr_video_proc();

int main()
{
	//sgl_clr_image_proc();
	sgl_clr_video_proc();
	//mu_clr_video_proc();
	
	cv::waitKey(0);
	return 0;
}


int sgl_clr_image_proc() {
	using namespace cv;
	double angle;
	char det_color;

	Mat image = imread("E:\\Code library\\USV-Competition\\USV\\test_materials\\test_2.jpg");
	cv::resize(image, image, cv::Size(1280, 720));
	colorDetecter color_det_img(image);

	if (color_det_img.process(image, colorDetecter::runMode::DEBUG, colorDetecter::clrMode::SGL, 30))
	{
		angle = color_det_img.get_angle();
		det_color = color_det_img.get_detectedColor();
		std::cout << "angle = " << angle << std::endl;
		std::cout << "detected color: " << det_color << std::endl;
		cv::imshow("result", image);
	}
	else
		std::cout << "No targert color detected!\n";
	waitKey(0);
	return 0;
}

int mu_clr_image_proc()
{
	using namespace cv;
	double angle;
	char det_color;

	Mat image = imread("E:\\Code library\\USV-Competition\\USV\\test_materials\\test_2.jpg");
	cv::resize(image, image, cv::Size(1280, 720));
	colorDetecter color_det_img(image);

	if (color_det_img.process(image, colorDetecter::runMode::DEBUG, colorDetecter::clrMode::SGL, 30))
	{
		angle = color_det_img.get_angle();
		det_color = color_det_img.get_detectedColor();
		std::cout << "angle = " << angle << std::endl;
		std::cout << "detected color: " << det_color << std::endl;
		cv::imshow("result", image);
	}
	else
		std::cout << "No targert color detected!\n";
	waitKey(0);
	return 0;
}

int sgl_clr_video_proc()
{
	std::string videoPath = "E:\\Code library\\USV-Competition\\USV\\test_materials\\video.mp4";
	cv::Mat srcImage;
	cv::Mat result;
	cv::namedWindow("Result");

	cv::VideoCapture cap(videoPath);
	if (!cap.isOpened())
	{
		std::cout << "Error opening video file£º" << videoPath << std::endl;
		return -1;
	}

	while (1)
	{
		//read a frame of the video
		cap >> srcImage;
		if (srcImage.empty())
			break;
		cv::resize(srcImage, srcImage, cv::Size(1280, 720));
		result = srcImage.clone();

		colorDetecter color_det_video(srcImage);
		if (color_det_video.process(result, colorDetecter::runMode::DEBUG, colorDetecter::clrMode::SGL, 30))
		{
			double angle = color_det_video.get_angle();
			char det_color = color_det_video.get_detectedColor();
			std::cout << "angle = " << angle << std::endl;
			std::cout << "detected color: " << det_color << std::endl;
		}
		else
			std::cout << "No targert color detected!\n";
		cv::imshow("Result", result);
		char c = cv::waitKey(33);
		if (c == 27)
			break;
	}
	cv::destroyWindow("Result");
	cap.release();

	return 0;
}



int mu_clr_video_proc()
{
	std::string videoPath = "E:\\Code library\\USV-Competition\\USV\\test_materials\\video.mp4";
	cv::Mat srcImage;
	cv::Mat result;
	cv::namedWindow("Result");

	cv::VideoCapture cap(videoPath);
	if (!cap.isOpened())
	{
		std::cout << "Error opening video file£º" << videoPath << std::endl;
		return -1;
	}

	while (1)
	{
		//read a frame of the video
		cap >> srcImage;
		if (srcImage.empty())
			break;
		cv::resize(srcImage, srcImage, cv::Size(1280, 720));
		result = srcImage.clone();

		colorDetecter color_det_video(srcImage);
		if (color_det_video.process(result, colorDetecter::runMode::DEBUG, colorDetecter::clrMode::MU, 30))
		{
			double * angle = color_det_video.get_mu_angle();
			char * det_color = color_det_video.get_mu_detectedColor();
			for (int i = 0; i < 3; i++)
			{
				std::cout << "detected color: " << det_color[i] << "  ";
				std::cout << "angle = " << angle[i] << "  ";
			}
			std::cout << std::endl;
		}
		else
			std::cout << "No targert color detected!\n";
		cv::imshow("Result", result);
		char c = cv::waitKey(33);
		if (c == 27)
			break;
	}
	cv::destroyWindow("Result");
	cap.release();
	return 0;
}
