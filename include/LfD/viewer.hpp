#ifndef __LFD_VIEWER_HPP__
#define __LFD_VIEWER_HPP__

#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pangolin/pangolin.h>

//generate point cloud with depth image
void gen_point_cloud(cv::Mat& color, cv::Mat& depth, double factor, Eigen::Matrix3d K, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);

//draw ellipsoids
void show_ellipsoid(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, Eigen::MatrixXd& ellipsoids, Eigen::MatrixXd& campose);

//convert Eigen::Matrix to OpenGlMatrix
void rtToOpenGTMatrix(Eigen::Matrix4d& T, pangolin::OpenGlMatrix& M);


Eigen::VectorXd evaluate_estimate(cv::Mat& depth, Eigen::VectorXd& bbs, double factor, Eigen::Matrix3d K, Eigen::MatrixXd& ellipsoids, Eigen::MatrixXd& campose);




#endif
