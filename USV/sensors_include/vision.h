#ifndef COLORDETECTER_H_
#define COLORDETECTER_H_
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\types_c.h>

//namespace cv{ using std::vector; }

class colorDetecter
{
private:
	cv::Mat image_;
	char detectedColor_;
	char detected_mu_Color_[4];
	double minLen_;
	int minh_, maxh_;
	int mins_, maxs_;
	int minv_, maxv_;
	double horizontal_fov_;
	double angle_;
	double mu_Angle_[3];
	cv::Size img_size_;

	//  Center_ will be calculated in the process function without targetColor param
	cv::Point2f Center_;

	// Center1_ and Center2_ are for the green balls detection
	// which will be calculated in the process function with targetColor param
	cv::Point2f Center1_;
	cv::Point2f Center2_;
	//for MU colMode
	cv::Point2f mu_Center_[3];
public:
	enum runMode {DEBUG, RELEASE};
	enum clrMode {SGL, MU};

private:
	void get_color_range(char targetColor);

	/**
	 *  @brief get mask of the target color
	 */
	void get_color_mask(char targetColor, cv::Mat & fhsv, cv::Mat & mask);

	void draw_result(cv::Mat & result, std::vector<std::vector<cv::Point>> & contours, int index,
		cv::Point center, float & radius);
	// overload
	void draw_result(cv::Mat & result, std::vector<std::vector<cv::Point>> & contours, float & radius1,
		float & radius2);

	void find_longest_contour(cv::Mat & mask, std::vector<std::vector<cv::Point>> & contours, double & maxLen,
		int & index);

	int find_maxLen_index(double * maxlen);

public:
	colorDetecter();        //defualt constructor

	colorDetecter(cv::Mat image, double minLen = 80, double horizontal_fov = 90);

	/**
	 *  @brief  Help you query the color information corresponding to the character
	 */
	void helpText();

	/**
	 *  @brief  set the horizontal fov of the camera
	 */
	void set_FOV(double horizontal_fov);

	/**
	 *  @brief  Gets the result of image processing
	 */
	char get_detectedColor() const;

	char * get_mu_detectedColor() const;

	/**
	 *  @brief  Get the orientation information of the target color
	 */
	double get_angle(int flag = 0);
	double * get_mu_angle();

	//void equalizeHist_clr(cv::Mat image);
	

	/**
 	 *  @brief  Check the target color in a picture. 
	    If only one green ball is detected, the results will be
		stored in Center1_. If two green ball are
		both detected, the results will be stored in both Center1_
		and Center2_. Other color's detection result will be stored
		in Center_.
		return value: -1 ！！ no target color detected or something wrong
					   0 ！！ target color detected (not green)
					   1 ！！ one green ball detected
					   2 ！！ two green ball detected
	 *  @param  targetColor: The target color that you want to check
	 *                  'H': red
	 *                  'L': blue
	 *                  'G': green
	 *                  'B': black
	 *  @param  result: a Mat data to show the origin image with the detection result
	 *  @param  runmode: release  debug
	 *  @param  minLen: The minimum value of the target color's perimeter
	 */
	int process_clr(char targetColor, cv::Mat & result, runMode runmode = RELEASE, double minLen = 50);

	/**
	 *  @brief  overload function: detect red、blue and black at the same time.The biggest one will
	    be seen as the detection result under SGN mode, but all of the three color detection result
		will be seen as the final result under the MU mode. Be used in the intermediate stages.
	 *  @param  result: a Mat data to show the origin image with the detection result
	 *  @param  runmode: RELEASE  DEBUG
	 *  @param  clrMode: control the color detection mode
					SGL: only the detection result of the biggest color will be stored in Center_
					 MU: detection result of each color(red、black、blue) will be stored by order in 
						 mu_Center_[3], once it can be detected
	 *  @param  minLen: The minimum value of the target color's perimeter
	 */
	int process_no_clr(cv::Mat & result, runMode runmode = RELEASE, clrMode clrmode = SGL, double minLen = 50);
};


class hazeMove
{
private:
	cv::Mat src_;
	cv::Mat dark_;
	cv::Mat te_;
	cv::Mat t_;
	int img_h_, img_w_;
	float outA_[3];		//Store atmospheric light intensity value of B、G、R channels
	int win_size_;		//window size of minimum filtering algorithm
	int r_;				//radius of guided filtering algorithm
	float eps_;			//A parameter that prevents the dividend from
						//being zero in the guided filtering algorithm
	float omega_;		//Parameter that determines defog intensity (0,1)
	float tx_;			//A parameter to prevent the image from shifting
						//to the white field in the dark channel algorithm

	/**
		@brief  Returns the corresponding subscript value of the array elements in ascending
		order, but does not change the array itself
	 */
	template<typename T>
	std::vector<int> argsort(const std::vector<T>& array);

	/**
		@brief  get the dark channel of an image
	 */
	cv::Mat DarkChannel(cv::Mat img) const;

	/**
		@brief  Calculating the atmospheric light intensity of an image(outA_[3])
	 */
	void AtmLight();

	/**
		@brief  Calculating the estimated transmission(te_)
	 */
	void TransmissionEstimate();
	
	/**
		@brief  guided filtering algorithm
	 */
	cv::Mat Guidedfilter(cv::Mat img_guid, cv::Mat te, int r, float eps) const;
	/**
		@brief  calculating transmission(t_) according to the estimated transmission(te_)
	 */
	void TransmissionRefine();

public:
	hazeMove();		//defualt constructor
	hazeMove(cv::Mat image);
	~hazeMove();

	/**
		@brief  defogging an image using the dark cahnnel algorithm��which is the
		most representative classic defogging algorithm proposed by He Kaiming
	 */
	cv::Mat Defogging();
	void ShowDark() { 
		cv::imshow("Dark", dark_);
	};
	void ShowTe() { 
		cv::imshow("te", te_);
	};
	void ShowT() { 
		cv::imshow("t", t_);
	};
	void ShowA() { 
		std::cout << outA_[0] << " " << outA_[1] << " " << outA_[2] << std::endl;
	};
	void SetParam(int win_size, int r, float eps, float omega, float tx);
};

#endif

