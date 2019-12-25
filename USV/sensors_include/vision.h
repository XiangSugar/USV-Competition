#ifndef COLORDETECTER_H_
#define COLORDETECTER_H_
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class colorDetecter
{
private:
	cv::Mat image_;
	char detectedColor_;
	double minLen_;
	int minh_, maxh_;
	int mins_, maxs_;
	int minv_, maxv_;
	double horizontal_fov_;
	double angle_;
	cv::Size img_size_;
	cv::Point2f Center_;

private:
	void dealColor(char targetColor);

public:
	colorDetecter();        //defualt constructor

	colorDetecter(cv::Mat image, double minLen = 80, double horizontal_fov = 90);

	/**
	 *  @brief  Help you query the color information corresponding to the character
	 */
	void helpText();


	void set_FOV(double horizontal_fov);

	/**
	 *  @brief  Gets the result of image processing
	 */
	char get_detectedColor() const;

	/**
	 *  @brief  Gets the orientation information of the target color
	 */
	double get_angle();

	/**
 	 *  @brief  Check the target color in a picture
	 *  @param  targetColor: The target color that you want to check
	 *                  'H': red
	 *                  'L': blue
	 *                  'G': green
	 *                  'B': black
	 *  @param  minLen: The minimum value of the target color's perimeter
	 *  @param  runMode: R！！Release  D！！Debug
	 */
	bool process(char targetColor, char runMode = 'R', double minLen = 80);
};
#endif

