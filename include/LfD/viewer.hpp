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


void gen_point_cloud(cv::Mat& color, cv::Mat& depth, double factor, Eigen::Matrix3d K, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);


void show_ellipsoid(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, Eigen::Matrix4d& ellipsoid, Eigen::MatrixXd& campose);

void rtToOpenGTMatrix(Eigen::Matrix4d& T, pangolin::OpenGlMatrix& M);




#endif
