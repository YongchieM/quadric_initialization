#include<iostream>
#include<opencv2/opencv.hpp>
#include "../include/myppf/headers.hpp"

using namespace std;
using namespace myppf;

int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr empty_scene(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr object(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal>::Ptr PCNormals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr sampledPC(new pcl::PointCloud<pcl::PointNormal>);
    
    loadOBJ(PCNormals, "/home/yongqi/project/data/scene/Plate.obj", 50000, true);
    //myppf::downSampling(PCNormals,sampledPC,7.5);
    //cout<<sampledPC->points.size()<<endl;
    
    /*Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::AngleAxisd rotation_vector(M_PI/2,Eigen::Vector3d(0,1,0));
    Eigen::Matrix3d rotation_matrix = rotation_vector.toRotationMatrix();
    T.block<3,3>(0,0) = rotation_matrix;
    T.topRightCorner<3,1>() = Eigen::Vector3d(50,25,50);
    cout<<T<<endl;
    myppf::transformPointCloud(sampledPC,T);
    pcl::io::savePLYFileASCII("cube_transformed.ply",*sampledPC);*/
    
    /*boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normals"));
    //viewer->addPointCloud<pcl::PointXYZRGB>(empty_scene,"cloud");
    viewer->addPointCloudNormals<pcl::PointNormal,pcl::PointNormal>(sampledPC,sampledPC,1,1,"normal");
    
    while(!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }*/
    
    //PPFDetector detector;
    //detector.trainModel(PCNormals);
    //detector.save_model("model.txt");
    //detector.test();
    
    
    //PPFDetector new_detector;
    //new_detector.load_model("model.txt");
    //new_detector.test();
  
    
    cout<<"hello"<<endl;
    return 0;
}
