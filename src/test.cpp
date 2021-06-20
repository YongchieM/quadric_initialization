#include <iostream>
#include <eigen3/Eigen/Dense>
#include "../include/LfD/estimate_ellipsoids.hpp"
#include "../include/LfD/helpers.hpp"
#include "../include/LfD/bbs_detect.hpp"
#include "../include/LfD/viewer.hpp"


using namespace std;

int main(int argc, char** argv)
{
    
    /*LfD::bbs_detector detector;
    detector.load_net();
    
    cv::Mat img;
    img = cv::imread("1341841280.114558.png");
    vector<pair<string,cv::Rect>> bbs;
    detector.detect(img,bbs);
    
    for(int i=0; i<bbs.size(); i++)
    {
        cout<<bbs[i].first<<endl;
        cout<<bbs[i].second.x<<" "<<bbs[i].second.y<<" "<<bbs[i].second.x+bbs[i].second.width<<" "<<bbs[i].second.y+bbs[i].second.height<<endl;
    }*/
    
    Eigen::MatrixXd camera,pose;
    LfD::load_data("tum/cabinet/camera.txt",camera);
    LfD::convertPose(camera,pose);
    
    
    Eigen::MatrixXd Cs,estCs,bbs,vis;
    Eigen::Matrix3d K;
    Eigen::Matrix<Eigen::Matrix4d,Eigen::Dynamic,1> estQs;
    
    
    LfD::load_data("tum/cabinet/bbs.txt",bbs);
    LfD::load_data("tum/cabinet/vis.txt",vis);
    LfD::load_intrinsics("tum/cabinet/intrinsics.txt",K);
    
    LfD::compute_estimates(bbs,K,pose,vis,Cs,estCs,estQs);
    
    /*LfD::eigMatrix C,estC,bb;
    
    int n_frames = vis.rows();
    const int cols = Cs.cols();
    C.resize(3,cols);
    estC.resize(3,cols);
    for(int i=0; i<n_frames; i++)
    {
        string img = "tum/cabinet/img"+to_string(i)+".png";
        C.row(0) = Cs.row(3*i);
        C.row(1) = Cs.row(3*i+1);
        C.row(2) = Cs.row(3*i+2);
        
        estC.row(0) = estCs.row(3*i);
        estC.row(1) = estCs.row(3*i+1);
        estC.row(2) = estCs.row(3*i+2);
        
        bb = bbs.row(i);
        
        LfD::plot_ellipse(img.c_str(),C,estC,bb);
    }*/
    
    double factor = 5000.0;
    cv::Mat color,depth;
    color = cv::imread("tum/cabinet/img0.png");
    depth = cv::imread("tum/cabinet/depth0.png",-1);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    
    Eigen::MatrixXd campose = pose.block<4,3>(0,0).transpose();
    gen_point_cloud(color,depth,factor,K,cloud);
    
    Eigen::MatrixXd Qs;
    
    /*Qs.resize(4*estQs.rows(),4);
    for(int i=0; i<estQs.rows(); i++)
        Qs.block<4,4>(4*i,0) = estQs[i];
    
    std::cout<<Qs<<std::endl;*/
    
    //Qs = LfD::getQStarFromBboxes(pose,bbs,K);
    
    show_ellipsoid(cloud,Qs,campose);
    

    
    
    return 0;
}

