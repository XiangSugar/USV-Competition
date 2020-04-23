#ifndef YOLODETECTOR_H_
#define YOLODETECTOR_H_

#include <opencv2/dnn.hpp>
#include <iostream>
#include <array>
using namespace cv::dnn;
using namespace std;

class yoloDetector
{
private:
	Net net_;
	float confThreshold_; // Confidence threshold
	float nmsThreshold_;  // Non-maximum suppression threshold
	int inpWidth_;  // Width of network's input image
	int inpHeight_; // Height of network's input image
	double horizontal_fov_;

	std::vector<string> classes_;

	std::vector<double> center_x_det_;
	std::vector<string> lables_det_;
	std::vector<int> size_boxs_det_;
public:
	enum runMode { DEBUG, RELEASE };

private:
	// Load names of classes
	void get_classes(const string &clsFile); //放入configration()函数中

	// Get the names of the output layers
	std::vector<std::string> get_outputs_names(const cv::dnn::Net &net);

	// Remove the bounding boxes with low confidence using non-maxima suppression
	int postprocess(cv::Mat& frame, const vector<cv::Mat> &out, runMode runmode);

	// Draw the predicted bounding box
	void draw_pred(int classId, float conf, int left, int top, int right,
		int bottom, cv::Mat& frame);
public:
	yoloDetector();
	//yoloDetector();
	~yoloDetector();

	void net_configration(const string &netCof, const string &modWeights,
		const string &clsFile);
	void set_param(float confThrd, float nmsThrd, int inpWd, int inpHg, double hor_fov);
	void update();

	int process(cv::Mat &frame, runMode runmode = RELEASE);

	bool get_target_ball(const char &target, int &indices) const;
	bool get_nearest_ball(int &indices, char &detectedColor) const;
	double get_angle(const int &indices, int img_width) const;
};
#endif // !YOLODETECTOR_H_