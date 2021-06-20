#include "../include/LfD/bbs_detect.hpp"

namespace LfD
{
    
bbs_detector::bbs_detector()
{
    confThreshold = 0.5;
    nmsThreshold = 0.4;
    inpWidth = 608;
    inpHeight = 608;
}

void bbs_detector::load_net()
{
    string classesFile = "/home/yongqi/project/data/yolo3/coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while(getline(ifs,line))
        classes.push_back(line);
    
    cv::String modelConfiguration = "/home/yongqi/project/data/yolo3/yolov3.cfg";
    cv::String modelWeight = "/home/yongqi/project/data/yolo3/yolov3.weights";
    net = cv::dnn::readNetFromDarknet(modelConfiguration,modelWeight);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
}
    
    
void bbs_detector::postprocess(cv::Mat& frame, const vector<cv::Mat>& outs, vector<pair<string,cv::Rect>>& bbs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    
    for(size_t i=0; i<outs.size(); i++)
    {
        float* data = (float*)outs[i].data;
        for(int j=0; j<outs[i].rows; j++,data+=outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5,outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores,0,&confidence,0,&classIdPoint);
            if(confidence>confThreshold)
            {
                int centerX = (int)(data[0]*frame.cols);
                int centerY = (int)(data[1]*frame.rows);
                int width = (int)(data[2]*frame.cols);
                int height = (int)(data[3]*frame.rows);
                int left = centerX-width/2;
                int top = centerY-height/2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left,top,width,height));
                
            }
        }
    }
    
    vector<int> indices;
    
    cv::dnn::NMSBoxes(boxes,confidences,confThreshold,nmsThreshold,indices);
    for(size_t i=0; i<indices.size(); i++)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        bbs.push_back(make_pair(classes[classIds[idx]],box));
        drawPred(classIds[idx],confidences[idx],box.x,box.y,box.x+box.width,box.y+box.height,frame);
    }
}



void bbs_detector::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    cv::rectangle(frame,cv::Point(left,top),cv::Point(right,bottom),cv::Scalar(255,178,50),3);
    
    string label = cv::format("%.2f",conf);
    if(!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId]+":"+label;
    }
    
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label,cv::FONT_HERSHEY_SIMPLEX,0.5,1,&baseLine);
    top = max(top,labelSize.height);
    cv::rectangle(frame,cv::Point(left,top-round(1.5*labelSize.height)),cv::Point(left+round(1.5*labelSize.width),top+baseLine),cv::Scalar(255,255,255),cv::FILLED);
    cv::putText(frame, label,cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75,cv::Scalar(0, 0, 0), 1);
    
}


vector<cv::String> bbs_detector::getOutputNames(const cv::dnn::Net& net){
    static vector<cv::String> names;
    if(names.empty()){
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<cv::String> layersNames = net.getLayerNames();
        
        names.resize(outLayers.size());
        for(size_t i =0;i<outLayers.size();i++){
            names[i] = layersNames[outLayers[i]-1];
        }
    }
    return names;
}


void bbs_detector::detect(cv::Mat& img, vector<pair<string,cv::Rect>>& bbs)
{
    cv::Mat blob;
    
    cv::dnn::blobFromImage(img,blob,1/255.0,cv::Size(inpWidth,inpHeight));
    net.setInput(blob);
    vector<cv::Mat> outs;
    net.forward(outs,getOutputNames(net));
    postprocess(img,outs,bbs);
    
    cv::Mat detectImg;
    img.convertTo(detectImg,CV_8U);
    cv::imshow("bounding boxes",detectImg);
    
    cv::waitKey(0);
    
}

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
}
