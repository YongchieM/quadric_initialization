#ifndef __LFD_HELPERS_HPP__
#define __LFD_HELPERS_HPP__

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <string>
#include <sstream>

#include "estimate_ellipsoids.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>

//#include <g2o/types/slam3d/se3quat.h>

namespace LfD
{


    
void load_data_py(const char* filename,eigMatrix& data)
{
    std::ifstream fin(filename);
    if(!fin.is_open())
        std::cerr<<"could not find data file"<<std::endl;
    
    std::vector<std::vector<double>> mat;
    std::string buff;
    while(getline(fin,buff))
    {
        char* s_input = (char*)buff.c_str();
        const char* split = ", ";
        char *p = strtok(s_input,split);
        double num;
        std::vector<double> vec;
        while(p!=NULL)
        {
            num = atof(p);
            vec.push_back(num);
            p = strtok(NULL,split);
        }
        mat.push_back(vec);
    }
    
    data.resize(mat.size(),mat[0].size());
    for(int i=0; i<mat.size(); i++)
        for(int j=0; j<mat[0].size(); j++)
            data(i,j) = mat[i][j];
    
    fin.close();
}

void load_data(const char* filename, eigMatrix& data)
{
    std::ifstream fin(filename);
    if(!fin.is_open())
        std::cerr<<"could not find data file"<<std::endl;
    
    int row_counter = 0;
    std::string line;
    double n;
    std::vector<double> first_row;
    while(std::getline(fin,line))
    {
        if(!line.empty())
        {
            std::stringstream ss(line);
            while(ss >> n)
            {
                first_row.push_back(n);
            }
            row_counter++;
            data.resize(100,first_row.size());
        }
        break;
    }
    for(int i=0; i<first_row.size(); i++)
        data(0,i) = first_row[i];
    
    while(std::getline(fin,line))
    {
        if(!line.empty())
        {
            std::stringstream ss(line);
            int colu = 0;
            while(ss >> n)
            {
                data(row_counter,colu) = n;
                colu++;
            }
            row_counter++;
            if(row_counter>=data.rows())
                data.conservativeResize(data.rows()*2,data.cols());
        }
    }
    
    fin.close();
    data.conservativeResize(row_counter,data.cols());
}



void load_intrinsics(const char* filename, Eigen::Matrix3d& K)
{
    std::ifstream fin(filename);
    if(!fin.is_open())
        std::cerr<<"could not find intrinsics file"<<std::endl;
    
    std::string buff;
    int row = 0;
    while(getline(fin,buff))
    {
        char* s_input = (char*)buff.c_str();
        const char* split = ", ";
        char *p = strtok(s_input,split);
        double num;
        int col = 0;
      
        while(p!=NULL)
        {
            num = atof(p);
            K(row,col) = num;
            p = strtok(NULL,split);
            col++;
        }
        row++;
    }
  
    fin.close();
}


void plot_ellipse(const char* filename,eigMatrix& Cs,eigMatrix& estCs,eigMatrix& bbs)
{
    cv::Mat img;
    img = cv::imread(filename);
    
    int n_obj = Cs.cols()/3;
    for(int i=0; i<n_obj; i++)
    {
        Eigen::Matrix3d C = Cs.block<3,3>(0,3*i);
        Eigen::Matrix3d estC = estCs.block<3,3>(0,3*i);
        Eigen::Vector4d bb = bbs.block<1,4>(0,4*i);
        Eigen::Vector2d centre,axes;
        Eigen::Matrix2d R;
        if(!C.isZero())
        {
            dual_ellipse_to_parameters(C,centre,axes,R);
            double rad = atan2(R(1,0),R(0,0));
            double deg = rad*180.0/M_PI;
            cv::Point ellipse_centre;
            cv::Size ellipse_axes;
            ellipse_centre.x = centre[0];
            ellipse_centre.y = centre[1];
            ellipse_axes.width = axes[0];
            ellipse_axes.height = axes[1];
            cv::ellipse(img,ellipse_centre,ellipse_axes,deg,0,360,cv::Scalar(0,255,0),2);
            
            if(!estC.isZero())
            {
                dual_ellipse_to_parameters(estC,centre,axes,R);
                rad = atan2(R(1,0),R(0,0));
                deg = rad*180.0/M_PI;
                ellipse_centre.x = centre[0];
                ellipse_centre.y = centre[1];
                ellipse_axes.width = axes[0];
                ellipse_axes.height = axes[1];
                cv::ellipse(img,ellipse_centre,ellipse_axes,deg,0,360,cv::Scalar(255,0,0),2);
            }
        }
        cv::rectangle(img,cv::Point(bb[0],bb[1]),cv::Point(bb[2],bb[3]),cv::Scalar(0,0,255),2);
    }
    
    cv::imshow("image",img);
    cv::waitKey(0);
}


Eigen::VectorXf rectToEig(std::vector<std::pair<std::string,cv::Rect>>& bbs)
{
    Eigen::VectorXf bbs_vec;
    bbs_vec.resize(4*bbs.size());
    
    
    for(int i=0; i<bbs.size(); i++)
    {
        bbs_vec.block<4,1>(4*i,0) = Eigen::Vector4f(bbs[i].second.x,bbs[i].second.y,bbs[i].second.x+bbs[i].second.width,bbs[i].second.y+bbs[i].second.height);
    }
    
    return bbs_vec;
    
}




Eigen::Matrix3Xd generateProjectionMatrix(Eigen::MatrixXd& campose_t, Eigen::Matrix3d& K)
{
    Eigen::MatrixXd campose_cw = campose_t.transpose();
    campose_cw.conservativeResize(campose_cw.rows()+1,campose_cw.cols());
    campose_cw.row(campose_cw.rows()-1)<<0,0,0,1;
    
    Eigen::Matrix3Xd identity_lefttop;
    identity_lefttop.resize(3,4);
    identity_lefttop.col(3) = Eigen::Vector3d(0,0,0);
    identity_lefttop.topLeftCorner<3,3>() = Eigen::Matrix3d::Identity(3,3);
    
    Eigen::Matrix3Xd proj_mat = K*identity_lefttop;
    
    proj_mat = proj_mat*campose_cw;
    
    return proj_mat;
}

Eigen::MatrixXd fromDetectionsToLines(Eigen::VectorXd &detections)
{
    double x1 = detections(0);
    double y1 = detections(1);
    double x2 = detections(2);
    double y2 = detections(3);
    
    Eigen::Vector3d line1(1,0,-x1);
    Eigen::Vector3d line2(0,1,-y1);
    Eigen::Vector3d line3(1,0,-x2);
    Eigen::Vector3d line4(0,1,-y2);
    
    Eigen::MatrixXd lines;
    lines.resize(3,4);
    
    lines.col(0) = line1;
    lines.col(1) = line2;
    lines.col(2) = line3;
    lines.col(3) = line4;
    
    return lines;
}

Eigen::MatrixXd getPlanesHomo(Eigen::MatrixXd& pose_mat, Eigen::MatrixXd& detection_mat, Eigen::Matrix3d& K)
{
    //std::assert(pose_mat.rows()/4==detection_mat.rows() && "Two matrics should match.");
    //std::assert(pose_mat.rows()/4>2 && "At least 3 measurements are required.");
    
    Eigen::MatrixXd planes_all(4,0);
    
    int nums = detection_mat.rows();
    for(int i=0; i<nums; i++)
    {
        Eigen::MatrixXd pose = pose_mat.block<4,3>(4*i,0);
        Eigen::VectorXd detection = detection_mat.row(i);
        
        if(detection(0)<1 && detection(1)<1 && detection(2)<1 && detection(3)<1)
            continue;
        
        Eigen::MatrixXd P = generateProjectionMatrix(pose,K);
        
        Eigen::MatrixXd lines = fromDetectionsToLines(detection);
        Eigen::MatrixXd planes = P.transpose()*lines;
        
        for(int m=0; m<planes.cols(); m++)
        {
            planes_all.conservativeResize(planes_all.rows(),planes_all.cols()+1);
            planes_all.col(planes_all.cols()-1)=planes.col(m);
        }
        
    }
    return planes_all;
}



Eigen::MatrixXd getVectorFromPlanesHomo(Eigen::MatrixXd& planes)
{
    int cols = planes.cols();
    
    Eigen::MatrixXd planes_vector(10,0);
    
    for(int i=0; i<cols; i++)
    {
        Eigen::VectorXd p = planes.col(i);
        Eigen::Matrix<double,10,1> v;
        
        v<<p(0)*p(0),2*p(0)*p(1),2*p(0)*p(2),2*p(0)*p(3),p(1)*p(1),2*p(1)*p(2),2*p(1)*p(3),p(2)*p(2),2*p(2)*p(3),p(3)*p(3);
        
        planes_vector.conservativeResize(planes_vector.rows(),planes_vector.cols()+1);
        planes_vector.col(planes_vector.cols()-1) = v;
    }
    
    return planes_vector;
}


Eigen::Matrix4d getQStarFromVectors(Eigen::MatrixXd& planeVecs)
{
    Eigen::MatrixXd A = planeVecs.transpose();
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeThinU|Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV();
    
    Eigen::VectorXd qj_hat = V.col(V.cols()-1);
    
    Eigen::Matrix4d Qstar;
    Qstar <<
            qj_hat(0),qj_hat(1),qj_hat(2),qj_hat(3),
            qj_hat(1),qj_hat(4),qj_hat(5),qj_hat(6),
            qj_hat(2),qj_hat(5),qj_hat(7),qj_hat(8),
            qj_hat(3),qj_hat(6),qj_hat(8),qj_hat(9);
            
    return Qstar;
}
    
    

Eigen::Matrix4d generate_ellipsoid_from_planes(Eigen::MatrixXd& pose,Eigen::MatrixXd& detection_mat,Eigen::Matrix3d& K)
{
    Eigen::MatrixXd pose_mat;
    convertPose(pose,pose_mat);
    
    Eigen::MatrixXd planesHomo = getPlanesHomo(pose_mat,detection_mat,K);
    
    
    //at least 9 planes are needed
    
    Eigen::MatrixXd planesVector = getVectorFromPlanesHomo(planesHomo);
    Eigen::Matrix4d Qstar = getQStarFromVectors(planesVector);
    
    return Qstar;
}



}

#endif
