// use_vision.cpp -- example of using the colorDetecter class

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "../sensors_include/vision.h"
#include <ctime>

int sgl_clr_image_proc();
int mu_clr_image_proc();
int sgl_clr_video_proc();
int mu_clr_video_proc();
int sgl_clr_video_proc_clr(char targetcolor);
int sgl_clr_image_proc_clr(char targetcolor);
void haze_move_test();

int main()
{
	//clock_t start = clock();

	//sgl_clr_image_proc();
	//mu_clr_image_proc();
	//sgl_clr_video_proc();
	mu_clr_video_proc();
	//sgl_clr_image_proc_clr('H');
	//sgl_clr_video_proc_clr('H');

	//haze_move_test();
	//std::cout << "total time: " << clock() - start << std::endl;
	
	//cv::waitKey(0);
	std::cin.get();
	return 0;
}


int sgl_clr_image_proc() {
	using namespace cv;
	double angle;
	char det_color;

	Mat image = imread("E:\\Code library\\USV-Competition\\USV\\test_materials\\test_4.jpg");
	cv::resize(image, image, cv::Size(1280, 720));
	colorDetecter color_det_img(image);

	if (color_det_img.process_no_clr(image, colorDetecter::runMode::DEBUG, colorDetecter::clrMode::SGL, 30))
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

	Mat image = imread("E:\\Code library\\USV-Competition\\USV\\test_materials\\test_4.jpg");
	cv::resize(image, image, cv::Size(1280, 720));
	colorDetecter color_det_img(image);

	if (color_det_img.process_no_clr(image, colorDetecter::runMode::DEBUG, colorDetecter::clrMode::MU, 30))
	{
		std::array<double, 3> angle = color_det_img.get_mu_angle();
		std::array<char, 4> det_color = color_det_img.get_mu_detectedColor();
		for (int i = 0; i < 3; i++)
		{
			std::cout << "detected color: " << det_color[i] << "  ";
			std::cout << "angle = " << angle[i] << std::endl;
		}
		//std::cout << std::endl;
	}
	else
		std::cout << "No targert color detected!\n";
	
	cv::imshow("result", image);

	waitKey(0);
	return 0;
}

int sgl_clr_image_proc_clr(char targetcolor)
{
	using namespace cv;
	double angle;
	char det_color;

	Mat image = imread("E:\\Code library\\USV-Competition\\USV\\test_materials\\test_4.jpg");
	cv::resize(image, image, cv::Size(1280, 720));
	colorDetecter color_det_img(image);

	if (0 == color_det_img.process_clr(targetcolor, image, colorDetecter::runMode::DEBUG, 30))
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

int sgl_clr_video_proc_clr(char targetcolor)
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
		//cv::resize(srcImage, srcImage, cv::Size(960, 540));
		//cv::resize(srcImage, srcImage, cv::Size(960, 540));
		result = srcImage.clone();

		colorDetecter color_det_video(srcImage);
		if (0 == color_det_video.process_clr(targetcolor, result, colorDetecter::runMode::DEBUG, 30))
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
		//cv::resize(srcImage, srcImage, cv::Size(960, 540));
		result = srcImage.clone();

		colorDetecter color_det_video(srcImage);
		if (color_det_video.process_no_clr(result, colorDetecter::runMode::DEBUG, colorDetecter::clrMode::SGL, 30))
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
		clock_t start = clock();
		cap >> srcImage;
		if (srcImage.empty())
			break;
		//cv::resize(srcImage, srcImage, cv::Size(1280, 720));
		result = srcImage.clone();

		colorDetecter color_det_video(srcImage);
		if (color_det_video.process_no_clr(result, colorDetecter::runMode::DEBUG, colorDetecter::clrMode::MU, 30))
		{
			std::array<double, 3> angle = color_det_video.get_mu_angle();
			std::array<char, 4> det_color = color_det_video.get_mu_detectedColor();
			for (int i = 0; i < 3; i++)
			{
				std::cout << "detected color: " << det_color[i] << "  ";
				std::cout << "angle = " << angle[i] << ",\t";
			}
			std::cout << std::endl;
		}
		else
			std::cout << "No targert color detected!\n";
		cv::imshow("Result", result);

		std::cout << "total time: " << clock() - start << std::endl;
		
		char c = cv::waitKey(10);
		if (c == 27)
			break;
		
	}
	cv::destroyWindow("Result");
	cap.release();
	return 0;
}

void haze_move_test()
{
	cv::Mat image = cv::imread("E:\\Code library\\USV-Competition\\USV\\test_materials\\defog_test2.jpg");
	//cv::resize(image, image, cv::Size(960, 540));
	cv::imshow("Image", image);
	clock_t start = clock();
	hazeMove haze_move(image);
	cv::Mat result = haze_move.Defogging();
	std::cout << "total time: " << clock() - start << std::endl;
	//haze_move.ShowA();
	//haze_move.ShowDark();
	//haze_move.ShowTe();
	haze_move.ShowT();

	cv::imshow("Result", result);

	cv::waitKey();
}
