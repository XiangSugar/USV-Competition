#include "../sensors_include/vision_yolo.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace cv::dnn;

yoloDetector::yoloDetector()
{
	confThreshold_ = 0.4;
	nmsThreshold_ = 0.4;
	inpWidth_ = 416;
	inpHeight_ = 416;
	horizontal_fov_ = 90;
}

yoloDetector::~yoloDetector() {}

void yoloDetector::get_classes(const string &clsFile)
{
	ifstream ifs(clsFile.c_str());
	string line;
	while (getline(ifs, line)) classes_.push_back(line);
}

void yoloDetector::net_configration(const string &netCof, const string &modWeights,
	const string &clsFile)
{
	net_ = readNetFromDarknet(netCof, modWeights);
	net_.setPreferableBackend(DNN_BACKEND_OPENCV);
	net_.setPreferableTarget(DNN_TARGET_CPU);

	get_classes(clsFile);
}

void yoloDetector::set_param(float confThrd, float nmsThrd, int inpWd, int inpHg,
	double hor_fov)
{
	confThreshold_ = confThrd;
	nmsThreshold_ = nmsThrd;
	inpWidth_ = inpWd;
	inpHeight_ = inpHg;
	horizontal_fov_ = hor_fov;
}

std::vector<std::string> yoloDetector::get_outputs_names(const cv::dnn::Net &net)
{
	static vector<string> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<string> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void yoloDetector::update()
{
	while (center_x_det_.size())
	{
		center_x_det_.pop_back();
		lables_det_.pop_back();
		size_boxs_det_.pop_back();
	}
}

int yoloDetector::process(cv::Mat &frame, runMode runmode)
{
	cv::Mat blob;
	blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth_, inpHeight_),
		cv::Scalar(0, 0, 0), true, false);

	//Sets the input to the network
	net_.setInput(blob);

	// Runs the forward pass to get output of the output layers
	vector<cv::Mat> outs;
	net_.forward(outs, get_outputs_names(net_));

	// Remove the bounding boxes with low confidence
	return postprocess(frame, outs, runmode);
}

// Remove the bounding boxes with low confidence using non-maxima suppression
int yoloDetector::postprocess(cv::Mat &frame, const vector<cv::Mat> &outs, runMode runmode)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<cv::Rect> boxes;

	vector<double> center_x;
	vector<int> size_boxs;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			int box_size;
			cv::Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold_)
			{
				//这个就是检测框的中心点
				int centerX = (int)(data[0] * frame.cols);
				if (DEBUG == runmode)
				{
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;
					boxes.push_back(cv::Rect(left, top, width, height));
					box_size = width + height;
				}
				center_x.push_back(centerX);
				size_boxs.push_back(box_size);
				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold_, nmsThreshold_, indices);
	int i;
	for (i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		lables_det_.push_back(classes_[classIds[idx]]);
		center_x_det_.push_back(center_x[idx]);
		size_boxs_det_.push_back(size_boxs[idx]);

		if (DEBUG == runmode)
		{
			cv::Rect box = boxes[idx];
			draw_pred(classIds[idx], confidences[idx], box.x, box.y,
				box.x + box.width, box.y + box.height, frame);
		}
	}
	if (0 == i)
	{
		cout << "No object detected!\n";
		return 0;
	}
	else
		return i;
}

// Draw the predicted bounding box
void yoloDetector::draw_pred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame)
{
	//Draw a rectangle displaying the bounding box
	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = cv::format("%.2f", conf);
	if (!classes_.empty())
	{
		CV_Assert(classId < (int)classes_.size());
		label = classes_[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	cv::rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)),
		cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
	cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
}

bool yoloDetector::get_target_ball(const char &target, int &indices) const
{
	int size = lables_det_.size();
	string targetball;
	switch (target)
	{
	case 'H':
		targetball = "redball";
		break;
	case 'B':
		targetball = "blackball";
		break;
	case 'L':
		targetball = "blueball";
		break;
	default:
		cout << "Error target color! ( In function get_target_ball() )" << endl;
		return false;
	}
	for (int i = 0; i < size; i++)
	{
		if (targetball == lables_det_[i])
		{
			indices = i;
			return true;
		}
	}
	cout << "No target object detected!\n";
	return false;
}

bool yoloDetector::get_nearest_ball(int &indices, char &detectedColor) const
{
	int size = lables_det_.size();
	if (size > 0)
	{
		int longest = 0;
		int i;
		for (i = 0; i < size; i++)
			if (longest < size_boxs_det_[i])
				longest = size_boxs_det_[i];
		indices = i - 1;
		string temp = lables_det_[i - 1];
		if ("redball" == temp)
			detectedColor = 'H';
		else if ("blackball" == temp)
			detectedColor = 'B';
		else
			detectedColor = 'L';
		return true;
	}
	else
	{
		detectedColor = 'N';
		return false;
	}
}

double yoloDetector::get_angle(const int &indices, int img_width) const
{
	double hw = img_width / 2.0;
	// The counterclockwise direction is positive
	double angle = horizontal_fov_ * (hw - center_x_det_[indices]) / (2 * hw);
	return angle;
}