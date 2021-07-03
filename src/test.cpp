#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "../include/LfD/estimate_ellipsoids.hpp"
#include "../include/LfD/helpers.hpp"
#include "../include/LfD/viewer.hpp"


using namespace std;

int main(int argc, char** argv)
{
    
    Eigen::MatrixXd pose,bbox;
    Eigen::Matrix3d K;
    
    
    LfD::load_data("tum/cabinet/bbox.txt",bbox);
    LfD::load_data("tum/cabinet/campose.txt",pose);
    LfD::load_intrinsics("tum/cabinet/intrinsics.txt",K);
    
    Eigen::Matrix4d Q;
    
    Eigen::MatrixXd selected_bbox = bbox.topRows(9);
    Eigen::MatrixXd selected_pose = pose.topRows(9);
    
    Q = LfD::generate_ellipsoid_from_ellipses(selected_pose,selected_bbox,K);
    //Q = LfD::generate_ellipsoid_from_planes(selected_pose,selected_bbox,K);
    
    cout<<"Q matrix is: "<<endl<<Q<<endl;
    
    Eigen::EigenSolver<Eigen::Matrix4d> es(Q);
    Eigen::Matrix4d D = es.pseudoEigenvalueMatrix();
    Eigen::Matrix4d V = es.pseudoEigenvectors();
    cout<<"eigen vlues are: "<<endl<<D<<endl;
    
    // visualisation
    double factor = 5000.0;
    cv::Mat color,depth;
    color = cv::imread("tum/cabinet/rgb/1341841281.5546.jpg");
    depth = cv::imread("tum/cabinet/depth/1341841281.5546.png",-1);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    
    gen_point_cloud(color,depth,factor,K,cloud);
    
    
    Eigen::MatrixXd pose_vis = pose.row(4);
    
    show_ellipsoid(cloud,Q,pose_vis);
    
    return 0;
}

