#ifndef __LFD_BBS_DETECT_HPP__
#define __LFD_BBS_DETECT_HPP__


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;

namespace LfD
{
    
class bbs_detector
{
    
public:

bbs_detector();

void load_net();

void postprocess(cv::Mat& frame,const vector<cv::Mat>& out, vector<pair<string,cv::Rect>>& bbs);

void drawPred(int classId,float conf,int left,int top,int right,int bottom,cv::Mat& frame);

vector<cv::String> getOutputNames(const cv::dnn::Net& net);

void detect(cv::Mat& img, vector<pair<string,cv::Rect>>& bbs);
    

private:
float confThreshold;
float nmsThreshold;
int inpWidth;
int inpHeight;
vector<string> classes;
cv::dnn::Net net;

};
    
}





#endif
