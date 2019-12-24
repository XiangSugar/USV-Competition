#ifndef COLORDETECTER_H_
#define COLORDETECTER_H_

class colorDetecter
{
private:
	cv::Mat image_;
	char detectedColor_;
	int minLen_;
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

	colorDetecter(cv::Mat image, int minLen, double horizontal_fov);

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
	 */
	bool process(char targetColor, int minLen);
};
#endif

