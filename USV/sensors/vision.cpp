#include "../sensors_include/vision.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <array>

colorDetecter::colorDetecter()      //defualt constructor
{
	minLen_ = 30;;
	horizontal_fov_ = 90;
	Center_ = cv::Point2f(0.0, 0.0);
	Center1_ = cv::Point2f(0.0, 0.0);
	Center2_ = cv::Point2f(0.0, 0.0);

	angle_ = 0.0;
	for (int i = 0; i < 3; i++) {
		mu_Angle_[i] = 0.0;
		mu_Center_[i] = cv::Point2f(0.0, 0.0);
		detected_mu_Color_[i] = 'N';
	}
	detectedColor_ = 'N';
	detected_mu_Color_[3] = '\0';
}

colorDetecter::~colorDetecter() {}

colorDetecter::colorDetecter(cv::Mat image, double minLen, double horizontal_fov)
{
	image_ = image;
	minLen_ = minLen;
	horizontal_fov_ = horizontal_fov;
	img_size_ = cv::Size(image.cols, image.rows);
	Center_ = cv::Point2f(0.0, 0.0);
	Center1_ = cv::Point2f(0.0, 0.0);
	Center2_ = cv::Point2f(0.0, 0.0);
	
	angle_ = 0.0;
	for (int i = 0; i < 3; i++) {
		mu_Angle_[i] = 0.0;
		mu_Center_[i] = cv::Point2f(0.0, 0.0);
		detected_mu_Color_[i] = 'N';
	}
	detectedColor_ = 'N';
	detected_mu_Color_[3] = '\0';
}

void colorDetecter::update_frame(cv::Mat &img)
{
	image_ = img;
	img_size_ = cv::Size(image_.cols, image_.rows);
	Center_ = cv::Point2f(0.0, 0.0);
	Center1_ = cv::Point2f(0.0, 0.0);
	Center2_ = cv::Point2f(0.0, 0.0);

	for (int i = 0; i < 3; i++) {
		mu_Center_[i] = cv::Point2f(0.0, 0.0);
		detected_mu_Color_[i] = 'N';
	}
	detectedColor_ = 'N';
	detected_mu_Color_[3] = '\0';
}

void colorDetecter::get_color_range(const char targetColor)
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
		maxv_ = 50;
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
		mins_ = 130;
		maxs_ = 255;
		minv_ = 50;
		maxv_ = 255;
		break;
	case 'r':	//red
		minh_ = 156;
		maxh_ = 179;
		mins_ = 130;
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
		mins_ = 130;
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

void colorDetecter::helpText() const
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

std::array<char, 4> colorDetecter::get_mu_detectedColor() const
{
	//TO DO
	return detected_mu_Color_;
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

const std::array<double, 3> colorDetecter::get_mu_angle()
{
	double hw = img_size_.width / 2.0;
	for (int i = 0; i < 3; i++)
		if (0.0 == mu_Center_[i].x)
			mu_Angle_[i] = -1000;
		else
			mu_Angle_[i] = (hw - mu_Center_[i].x) / img_size_.width * horizontal_fov_;

	return mu_Angle_;
}

//void colorDetecter::equalizeHist_clr(cv::Mat image)
//{}

void colorDetecter::get_color_mask(const char targetColor, cv::Mat & fhsv, cv::Mat & mask)
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
	//cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);
}

void colorDetecter::draw_result(cv::Mat & result, std::vector<std::vector<cv::Point>> & contours,
	int index, cv::Point center, float & radius, const char color) const
{
	if ('H' == color)
	{
		cv::drawContours(result, contours, index, cv::Scalar(0, 0, 255), 1, 8);
		cv::circle(result, center, (int)radius, cv::Scalar(0, 0, 255), 2);
	}
	else if ('L' == color)
	{
		cv::drawContours(result, contours, index, cv::Scalar(255, 0, 0), 1, 8);
		cv::circle(result, center, (int)radius, cv::Scalar(255, 0, 0), 2);
	}
	else if ('B' == color)
	{
		cv::drawContours(result, contours, index, cv::Scalar(0, 0, 0), 1, 8);
		cv::circle(result, center, (int)radius, cv::Scalar(0, 0, 0), 2);
	}
	else if ('G' == color)
	{
		cv::drawContours(result, contours, index, cv::Scalar(0, 255, 0), 2, 8);
		cv::circle(result, center, (int)radius, cv::Scalar(0, 255, 0), 1);
	}
	else
		std::cout << "ERROR: aim color error!" << std::endl;
}

void colorDetecter::draw_result(cv::Mat & result, std::vector<std::vector<cv::Point>> & contours,
	float & radius1, float & radius2) const
{
	cv::drawContours(result, contours, 0, cv::Scalar(0, 255, 0), 2, 8);
	cv::circle(result, (cv::Point)Center1_, (int)radius1, cv::Scalar(0, 0, 255), 1);
	cv::drawContours(result, contours, 1, cv::Scalar(0, 255, 0), 2, 8);
	cv::circle(result, (cv::Point)Center2_, (int)radius2, cv::Scalar(0, 0, 255), 1);
}

void colorDetecter::find_longest_contour_index(cv::Mat & mask, std::vector<std::vector<cv::Point>> & contours,
	double & maxLen, int & index) const
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

int colorDetecter::find_maxLen_index(std::array<double, 3> &maxlen) const
{
	int index = 0;
	for (int i = 0; i < 3; i++)
		if (maxlen[i] > maxlen[index])
			index = i;
	return index;
}

void colorDetecter::contours_sizes_sort(std::vector<double> &contour_size, int size,
								       std::vector<std::vector<cv::Point>> &contours) const
{
	std::vector<cv::Point> temp_Contour;
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
}

int colorDetecter::process_clr(const char targetColor, cv::Mat & result, runMode runmode, double minLen)
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

	cv::Mat img_Blur_fhsv;
	cv::GaussianBlur(image_, img_Blur_fhsv, cv::Size(3, 3), 0, 0);
	//cv::Mat fhsv;
	cv::cvtColor(img_Blur_fhsv, img_Blur_fhsv, COLOR_BGR2HSV);
	cv::Mat mask;
	get_color_mask(targetColor, img_Blur_fhsv, mask);

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
			// sort by contour's length in descending order
			contours_sizes_sort(contour_size, size, contours);

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
				state = -1;
		}
		else
			state = -1;

		if (DEBUG == runmode)
		{
			//cv::Mat result = cv::Mat::zeros(img_size_, CV_8UC3);
			if (2 == state)
				draw_result(result, contours, radius1, radius2);
			else if (1 == state)
				draw_result(result, contours, 0, Center1_, radius1, targetColor);
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
		find_longest_contour_index(mask, contours, maxLen, index);
		if (maxLen < minLen_)
		{
			detectedColor_ = 'N';	// find no target color
			return -1;
		}
		else
		{
			float radius;
			cv::minEnclosingCircle(contours[index], Center_, radius);
			detectedColor_ = targetColor;

			if (DEBUG == runmode)
			{
				draw_result(result, contours, index, Center_, radius, targetColor);
				/* for debug
				cv::imshow("mask", mask);
				*/
			}
			return 0;
		}
	}
}


int colorDetecter::process_no_clr(cv::Mat & result, runMode runmode, clrMode clrmode, double minLen)
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
	int index = 0;
	double maxLen_H = 0;
	double maxLen_B = 0;
	double maxLen_L = 0;
	int index_H = 0;
	int index_B = 0;
	int index_L = 0;
	cv::Mat fhsv;
	cv::Mat mask_H;
	cv::Mat mask_B;
	cv::Mat mask_L;
	std::vector<std::vector<cv::Point>> contours_H;
	std::vector<std::vector<cv::Point>> contours_B;
	std::vector<std::vector<cv::Point>> contours_L;
	cv::cvtColor(image_, fhsv, COLOR_BGR2HSV);

	get_color_mask('H', fhsv, mask_H);
	find_longest_contour_index(mask_H, contours_H, maxLen_H, index_H);
	get_color_mask('B', fhsv, mask_B);
	find_longest_contour_index(mask_B, contours_B, maxLen_B, index_B);
	get_color_mask('L', fhsv, mask_L);
	find_longest_contour_index(mask_L, contours_L, maxLen_L, index_L);

	if (SGL == clrmode)
	{
		std::array<double, 3> maxlen = { maxLen_H, maxLen_B, maxLen_L };
		index = find_maxLen_index(maxlen);
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
			if (DEBUG == runmode)
			{
				if (0 == index)
					draw_result(result, contours_H, index_H, Center_, radius, detectedColor_);
				else if (1 == index)
					draw_result(result, contours_B, index_B, Center_, radius, detectedColor_);
				else if (2 == index)
					draw_result(result, contours_L, index_L, Center_, radius, detectedColor_);
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
	else if (MU == clrmode)
	{
		std::array<std::vector<std::vector<cv::Point>>, 3> contours_set = { contours_H, contours_B, contours_L };
		std::array<float, 3> radius_set  = { 0, 0, 0 };
		std::array<double, 3> maxLen_set = { maxLen_H, maxLen_B, maxLen_L };
		std::array<int, 3> index_set     = { index_H, index_B, index_L };
		std::array<char, 3> color_set    = { 'H', 'B', 'L' };
		int count = 0;
		
		int i = 0;
		while (i < 3)
		{
			if (maxLen_set[i] < minLen_)
				detected_mu_Color_[i] = 'N';
			else
			{
				cv::minEnclosingCircle(contours_set[i][index_set[i]], mu_Center_[i], radius_set[i]);
				detected_mu_Color_[i] = color_set[i];
				count++;
			}
			i++;
		}
		if (count)
		{
			if (DEBUG == runmode)
			{
				int i = 0;
				while (i < 3)
				{
					if (maxLen_set[i] > minLen_)
						draw_result(result, contours_set[i], index_set[i], mu_Center_[i],
							radius_set[i], color_set[i]);
					i++;
				}
			
				/* for debug
				cv::imshow("mask_H", mask_H);
				cv::imshow("mask_B", mask_B);
				cv::imshow("mask_L", mask_L);
				*/
			}
			return 1;
		}
		else
		{
			std::cout << "No color detected!" << std::endl;
			return 0;
		}
	}
	else
	{
		std::cout << "Error: clrMode should be SGL or MU!" << std::endl;
		return 0;
	}	
}


// class hazeMove

hazeMove::hazeMove()
{
	for (int i = 0; i < 3; i++)
		outA_[i] = float(0.0);
	win_size_ = 15;
	r_ = 60;
	eps_ = 0.001;
	omega_ = 0.95;
	tx_ = 0.1;
}
hazeMove::hazeMove(cv::Mat image)
{
	src_ = image;
	img_h_ = image.rows;
	img_w_ = image.cols;
	dark_ = cv::Mat(img_h_, img_w_, CV_32FC1);
	te_ = cv::Mat(img_h_, img_w_, CV_32FC1);
	t_ = cv::Mat(img_h_, img_w_, CV_32FC1);
	for (int i = 0; i < 3; i++)
		outA_[i] = float(0.0);
	win_size_ = 25;
	r_ = 60;
	eps_ = 0.001;
	omega_ = 0.95;	//0.95
	tx_ = 0.1;
}
hazeMove::~hazeMove()
{
}

template<typename T> 
std::vector<int> hazeMove::argsort(const std::vector<T>& array)
{
	const int array_len(array.size());
	std::vector<int> array_index(array_len, 0);
	for (int i = 0; i < array_len; ++i)
		array_index[i] = i;

	std::sort(array_index.begin(), array_index.end(),
		[&array](int pos1, int pos2) {return (array[pos1] < array[pos2]); });

	return array_index;
}

cv::Mat hazeMove::DarkChannel(cv::Mat &img) const
{
	std::vector<cv::Mat> chanels(3);
	split(img, chanels);

	//求RGB三通道中的最小像像素值
	cv::Mat minChannel = (cv::min)((cv::min)(chanels[0], chanels[1]), chanels[2]);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(win_size_, win_size_));

	cv::Mat dark(img_h_, img_w_, CV_32FC1);
	cv::erode(minChannel, dark, kernel);	//图像腐蚀(实际上就是何凯明算法中的最小值滤波)
	return dark;
}
void hazeMove::AtmLight()
{
	int imgSize = img_h_ * img_w_;

	//将暗图像和原图转为列向量
	std::vector<int> darkVector = dark_.reshape(1, imgSize);
	cv::Mat src = src_;
	//将改变原有的数据，因此用原图的copy版来运算，防止原图被修改
	cv::Mat srcVector = src.reshape(3, imgSize); 
	//这个类型转换很有必要，前期就是在这里踩坑，很久才跳出来
	srcVector.convertTo(srcVector, CV_32FC3);

	//按照亮度的大小取前0.1%的像素（亮度高）
	int numpx = int(cv::max(floor(imgSize / 1000), 1.0));
	std::vector<int> indices = argsort(darkVector);
	std::vector<int> dstIndices(indices.begin() + (imgSize - numpx), indices.end());

	for (int i = 0; i < numpx; ++i)
	{
		outA_[0] += srcVector.at<cv::Vec3f>(dstIndices[i], 0)[0];
		outA_[1] += srcVector.at<cv::Vec3f>(dstIndices[i], 0)[1];
		outA_[2] += srcVector.at<cv::Vec3f>(dstIndices[i], 0)[2];
	}

	outA_[0] = cv::min(outA_[0] / numpx, float(225.0));
	outA_[1] = cv::min(outA_[1] / numpx, float(225.0));
	outA_[2] = cv::min(outA_[2] / numpx, float(225.0));
}
void hazeMove::TransmissionEstimate()
{
	cv::Mat imgA = cv::Mat::zeros(img_h_, img_w_, CV_32FC3);
	cv::Mat img = src_;

	//必须进行类型转换
	img.convertTo(img, CV_32FC3);
	std::vector<cv::Mat> chanels(CV_32FC1);
	split(img, chanels);
	for (int i = 0; i < 3; ++i)
		chanels[i] = chanels[i] / outA_[i];

	cv::merge(chanels, imgA);
	te_ = 1 - omega_ * DarkChannel(imgA);	//计算透射率预估值
}

cv::Mat hazeMove::Guidedfilter(cv::Mat img_guid, cv::Mat te, int r, float eps) const
{
	cv::resize(img_guid, img_guid, cv::Size(320,180));
	cv::resize(te, te, cv::Size(320,180));

	cv::Mat meanI, meanT, meanIT, meanII, meanA, meanB;
	cv::boxFilter(img_guid, meanI, CV_32F, cv::Size(r, r));
	cv::boxFilter(te, meanT, CV_32F, cv::Size(r, r));
	cv::boxFilter(img_guid.mul(te), meanIT, CV_32F, cv::Size(r, r));
	cv::Mat covIT = meanIT - meanI.mul(meanT);

	boxFilter(img_guid.mul(img_guid), meanII, CV_32F, cv::Size(r, r));
	cv::Mat varI = meanII - meanI.mul(meanI);

	cv::Mat a = covIT / (varI + eps);
	cv::Mat b = meanT - a.mul(meanI);
	boxFilter(a, meanA, CV_32F, cv::Size(r, r));
	boxFilter(b, meanB, CV_32F, cv::Size(r, r));

	cv::Mat t = meanA.mul(img_guid) + meanB;

	cv::resize(t, t, cv::Size(img_w_, img_h_));

	return t;
}

void hazeMove::TransmissionRefine()
{
	cv::Mat gray;
	cv::cvtColor(src_, gray, CV_BGR2GRAY);
	gray.convertTo(gray, CV_32F);
	gray /= 255;

	t_ = Guidedfilter(gray, te_, r_, eps_);
}

cv::Mat hazeMove::Defogging()
{
	dark_ = DarkChannel(src_);
	AtmLight();
	TransmissionEstimate();
	TransmissionRefine();

	cv::Mat dst = cv::Mat::zeros(img_h_, img_w_, CV_32FC3);
	cv::Mat t = (cv::max)(t_, tx_);				//设置阈值当投射图t 的值很小时，会导致图像整体向白场过度
	
	cv::Mat srcImg;
	src_.convertTo(srcImg, CV_32F);
	
	std::vector<cv::Mat> chanels;
	split(srcImg, chanels);
	for (int i = 0; i < 3; ++i)
		chanels[i] = (chanels[i] - outA_[i]) / t + outA_[i];
	merge(chanels, dst);
	
	//类型转换很重要
	dst.convertTo(dst, CV_8UC3);
	return dst;
}

void hazeMove::SetParam(int win_size, int r, float eps, float omega, float tx)
{
	win_size_ = win_size;
	r_ = r;
	eps_ = eps;
	omega_ = omega;
	tx_ = tx;
}
