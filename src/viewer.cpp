#include "../include/LfD/viewer.hpp"
#include "../include/LfD/estimate_ellipsoids.hpp"

void gen_point_cloud(cv::Mat& color, cv::Mat& depth, double factor, Eigen::Matrix3d K, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
    const double cx = K(0,2);
    const double cy = K(1,2);
    const double fx = K(0,0);
    const double fy = K(1,1);
    
    
    for(int m=0; m<depth.rows; m++)
        for(int n=0; n<depth.cols; n++)
        {
            ushort d = depth.ptr<ushort>(m)[n];
            if(d==0)
                continue;
            
            pcl::PointXYZRGBA p;
            p.z = double(d)/factor;
            p.x = (n-cx)*p.z/fx;
            p.y = (m-cy)*p.z/fy;
            
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
            
            cloud->points.push_back(p);
        }
        
}

void show_ellipsoid(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, Eigen::MatrixXd& ellipsoids, Eigen::MatrixXd& campose)
{
    Eigen::Matrix3d R;
    Eigen::Vector3d t, axis;
    
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = campose.block<3,3>(0,0);
    T.topRightCorner<3,1>() = campose.topRightCorner<3,1>();
    
    int obj = ellipsoids.rows()/4;
    
    pangolin::CreateWindowAndBind("point cloud viewer",1024,768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,100),
        pangolin::ModelViewLookAt(0,-0.1,-1.8,0,0,0,0.0,-1.0,0.0)
                                );
    pangolin::View &d_cam = pangolin::CreateDisplay().SetBounds(0.0,1.0,pangolin::Attach::Pix(175),1.0,-1024.0f/768.0f).SetHandler(new pangolin::Handler3D(s_cam));
    
    while(pangolin::ShouldQuit()==false)
    {
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(0.0f,0.0f,0.0f,0.0f);
        glPointSize(2);
        glBegin(GL_POINTS);
        for(auto &p:cloud->points)
        {
            glColor3f((float)p.r/255.0,(float)p.g/255.0,(float)p.b/255.0);
            glVertex3d(p.x,p.y,p.z);
        }
        glEnd();
        
        for(int i=0; i<obj; i++)
        {
        glPushMatrix();
        glLineWidth(3);
        glColor3f(0.0f,0.0f,1.0f);
        
       
            LfD::dual_quadric_to_ellipsoid_parameters(ellipsoids.block<4,4>(4*i,0),t,axis,R);
            Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
            pose.block<3,3>(0,0) = R;
            pose.topRightCorner<3,1>() = t;
            
            Eigen::Matrix4d new_pose = T*pose;
            pangolin::OpenGlMatrix Twm;
            rtToOpenGTMatrix(new_pose,Twm);
            
            GLUquadricObj *pObj;
            pObj = gluNewQuadric();
            gluQuadricDrawStyle(pObj,GLU_LINE);
            
            glMultMatrixd(Twm.m);
            glScaled(axis[0],axis[1],axis[2]);
            
            gluSphere(pObj,1.0,26,13);
        
        
        glPopMatrix();
        }
        
        pangolin::FinishFrame();
        usleep(5000);
    }
    return;
    
    
}

void rtToOpenGTMatrix(Eigen::Matrix4d& T, pangolin::OpenGlMatrix& M)
{

    M.m[0] = T(0,0);
    M.m[1] = T(1,0);
    M.m[2] = T(2,0);
    M.m[3] = 0.0;
    
    M.m[4] = T(0,1);
    M.m[5] = T(1,1);
    M.m[6] = T(2,1);
    M.m[7] = 0.0;
    
    M.m[8] = T(0,2);
    M.m[9] = T(1,2);
    M.m[10] = T(2,2);
    M.m[11] = 0.0;
    
    M.m[12] = T(0,3);
    M.m[13] = T(1,3);
    M.m[14] = T(2,3);
    M.m[15] = 1.0;
}

Eigen::VectorXd evaluate_estimate(cv::Mat& depth, Eigen::VectorXd& bbs, double factor, Eigen::Matrix3d K, Eigen::MatrixXd& ellipsoids, Eigen::MatrixXd& campose)
{
    const double cx = K(0,2);
    const double cy = K(1,2);
    const double fx = K(0,0);
    const double fy = K(1,1);
    
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = campose.block<3,3>(0,0);
    T.topRightCorner<3,1>() = campose.topRightCorner<3,1>();
    
    int obj_num = bbs.rows()/4;
    Eigen::MatrixXd centres;
    centres.resize(4,obj_num);
    centres.setZero();
    
    
    for(int i=0; i<obj_num; i++)
    {
        Eigen::VectorXd bb = bbs.block<4,1>(4*i,0);
        int u = (bb[0]+bb[2])/2;
        int v = (bb[1]+bb[3])/2;
        ushort d = depth.ptr<ushort>(v)[u];
        
        if(d == 0)
            continue;
        
        
        double z = (double)d/factor;
        double x = (u-cx)*z/fx;
        double y = (u-cy)*z/fy;
        
        centres.col(i)<<x,y,z,1;
    }
    
    Eigen::MatrixXd centres_global = T.inverse()*centres;
    
    cout<<centres_global<<endl;
    
    Eigen::VectorXd dis;
    dis.resize(obj_num);
    
    for(int i=0; i<obj_num; i++)
    {
        Eigen::Matrix4d Q = ellipsoids.block<4,4>(4*i,0);
        Eigen::Vector3d t = -Q.topRightCorner<3,1>();
        Eigen::Vector3d diff = centres_global.block<3,1>(0,i)-t;
        
        //cout<<diff<<endl;
        //cout<<"---------"<<endl;
        
        dis[i] = diff.norm();
    }
    
    return dis;
        
}

