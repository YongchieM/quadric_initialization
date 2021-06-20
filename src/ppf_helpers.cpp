#include "../include/myppf/headers.hpp"

namespace myppf
{
    
void loadPLY(pcl::PointCloud<pcl::PointXYZ>::Ptr PC, const char* filename)
{
    ifstream inFile(filename, ios::in | ios::binary);
    if (!inFile)
    {
        cerr << "can not open the file" << endl;
        return;
    }

    int numPts;
    std::string line;
    std::string num_flag = "element vertex ";
    int pos = num_flag.size();

    while (!inFile.eof())
    {
        getline(inFile, line);

        if (line.find(num_flag) != std::string::npos)
        {
            std::string num = line.substr(pos);
            numPts = atoi(num.c_str());
            cout << "There are " << numPts << " points" << endl;
        }

        if (line == "end_header")
            break;
    }

    float position[3];
    unsigned char color[3];

    PC->width = 1920;
    PC->height = 1200;
    PC->resize(PC->width*PC->height);
    PC->is_dense = false;
    pcl::PointXYZ point;

    int n = 0;
    
    while(n < numPts)
    {
        inFile.read((char*)&position, sizeof(position));
        if(!isinf(position[2]) && !isnan(position[2]))
        {
            point.x = position[0]/1000.0f;
            point.y = position[1]/1000.0f;
            point.z = position[2]/1000.0f;
        }
        else
        {
            point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
        }
        PC->points[n] = point;
        inFile.ignore(sizeof(color));
        n++;
    }
    
    inFile.close();
}

void loadPLYRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr PC, const char* filename)
{
    ifstream inFile(filename, ios::in | ios::binary);
    if (!inFile)
    {
        cerr << "can not open the file" << endl;
        return;
    }

    int numPts;
    std::string line;
    std::string num_flag = "element vertex ";
    int pos = num_flag.size();

    while (!inFile.eof())
    {
        getline(inFile, line);

        if (line.find(num_flag) != std::string::npos)
        {
            std::string num = line.substr(pos);
            numPts = atoi(num.c_str());
            //cout << "There are " << numPts << " points" << endl;
        }

        if (line == "end_header")
            break;
    }

    float position[3];
    unsigned char color[3];

    PC->width = 1920;
    PC->height = 1200;
    PC->resize(PC->width*PC->height);
    PC->is_dense = false;
    pcl::PointXYZRGB point;

    int n = 0;
    while(n<numPts)
    {
        inFile.read((char*)&position, sizeof(position));
        inFile.read((char*)&color, sizeof(color));
        if(!isinf(position[2]) && !isnan(position[2]))
        {
            point.x = position[0]/1000.0f;
            point.y = position[1]/1000.0f;
            point.z = position[2]/1000.0f;
            std::uint8_t r = color[0], g = color[1], b = color[2];    
            std::uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
            point.rgb = *reinterpret_cast<float*>(&rgb);
        }
        else
        {
            point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
        }
        PC->points[n] = point;
        n++;
    }
    inFile.close();
}

//sampling on the obj surface and compute normals
inline double uniform_deviate (int seed)
{
    double ran = seed * (1.0 / (RAND_MAX + 1.0));
    return ran;
}
 
inline void randomPointTriangle (float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3, Eigen::Vector3f& p)
{
    float r1 = static_cast<float> (uniform_deviate (rand ()));
    float r2 = static_cast<float> (uniform_deviate (rand ()));
    float r1sqr = std::sqrt (r1);
    float OneMinR1Sqr = (1 - r1sqr);
    float OneMinR2 = (1 - r2);
    a1 *= OneMinR1Sqr;
    a2 *= OneMinR1Sqr;
    a3 *= OneMinR1Sqr;
    b1 *= OneMinR2;
    b2 *= OneMinR2;
    b3 *= OneMinR2;
    c1 = r1sqr * (r2 * c1 + b1) + a1;
    c2 = r1sqr * (r2 * c2 + b2) + a2;
    c3 = r1sqr * (r2 * c3 + b3) + a3;
    p[0] = c1;
    p[1] = c2;
    p[2] = c3;
    //p[3] = 0;
}
 
inline void randPSurface (vtkPolyData * polydata, std::vector<double> * cumulativeAreas, double totalArea, Eigen::Vector3f& p, bool calcNormal, Eigen::Vector3f& n)
{
    float r = static_cast<float> (uniform_deviate (rand ()) * totalArea);
 
    std::vector<double>::iterator low = std::lower_bound (cumulativeAreas->begin (), cumulativeAreas->end (), r);
    vtkIdType el = vtkIdType (low - cumulativeAreas->begin ());
 
    double A[3], B[3], C[3];
    vtkIdType npts = 0;
    vtkIdType *ptIds = NULL;
    polydata->GetCellPoints (el, npts, ptIds);
    polydata->GetPoint (ptIds[0], A);
    polydata->GetPoint (ptIds[1], B);
    polydata->GetPoint (ptIds[2], C);
    if (calcNormal)
    {
        // OBJ: Vertices are stored in a counter-clockwise order by default
        Eigen::Vector3f v1 = Eigen::Vector3f (A[0], A[1], A[2]) - Eigen::Vector3f (C[0], C[1], C[2]);
        Eigen::Vector3f v2 = Eigen::Vector3f (B[0], B[1], B[2]) - Eigen::Vector3f (C[0], C[1], C[2]);
        n = v1.cross (v2);
        n.normalize ();
    }
    randomPointTriangle (float (A[0]), float (A[1]), float (A[2]),
                         float (B[0]), float (B[1]), float (B[2]),
                         float (C[0]), float (C[1]), float (C[2]), p);
}
 
void uniform_sampling (vtkSmartPointer<vtkPolyData> polydata, size_t n_samples, bool calc_normal, pcl::PointCloud<pcl::PointNormal>& cloud_out, const bool flipViewpoint)
{
    polydata->BuildCells ();
    vtkSmartPointer<vtkCellArray> cells = polydata->GetPolys ();
 
    double p1[3], p2[3], p3[3], totalArea = 0;
    std::vector<double> cumulativeAreas (cells->GetNumberOfCells (), 0);
    size_t i = 0;
    vtkIdType npts = 0, *ptIds = NULL;
    for (cells->InitTraversal (); cells->GetNextCell (npts, ptIds); i++)
    {
        polydata->GetPoint (ptIds[0], p1);
        polydata->GetPoint (ptIds[1], p2);
        polydata->GetPoint (ptIds[2], p3);
        totalArea += vtkTriangle::TriangleArea (p1, p2, p3);
        cumulativeAreas[i] = totalArea;
    }
 
    cloud_out.points.resize (n_samples);
    cloud_out.width = static_cast<pcl::uint32_t> (n_samples);
    cloud_out.height = 1;
 
    Eigen::Vector3d viewpoint(500.0,500.0,500.0);
    for (i = 0; i < n_samples; i++)
    {
        Eigen::Vector3f p;
        Eigen::Vector3f n;
        randPSurface (polydata, &cumulativeAreas, totalArea, p, calc_normal, n);
        cloud_out.points[i].x = p[0];
        cloud_out.points[i].y = p[1];
        cloud_out.points[i].z = p[2];
        Eigen::Vector3d point(p[0],p[1],p[2]);
        Eigen::Vector3d normal(n[0],n[1],n[2]);
        if(flipViewpoint)
            flipNormalViewpoint(point,viewpoint,normal);
        if (calc_normal)
        {
            cloud_out.points[i].normal_x = normal[0];
            cloud_out.points[i].normal_y = normal[1];
            cloud_out.points[i].normal_z = normal[2];
        }
    }
}

void loadOBJ(pcl::PointCloud<pcl::PointNormal>::Ptr PC, const char* filename, const int samples, const bool flipViewpoint=true)
{
    int SAMPLE_POINTS_ = samples;
    bool with_normals = true;

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFileOBJ(filename,mesh);
    pcl::io::mesh2vtk(mesh,polydata);
    
    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
#if VTK_MAJOR_VERSION < 6
    triangleFilter->SetInput(plydata);
#else
    triangleFilter->SetInputData(polydata);
#endif
    
    
    vtkSmartPointer<vtkPolyDataMapper> triangleMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    triangleMapper->SetInputConnection(triangleFilter->GetOutputPort());
    triangleMapper->Update();
    polydata = triangleMapper->GetInput();
    
    uniform_sampling(polydata,SAMPLE_POINTS_,with_normals,*PC, flipViewpoint);
}


void addToClusters(std::vector<std::vector<pcl::PointNormal>>& clusters, pcl::PointNormal point, const double threshold)
{
    Eigen::Vector3f average_normal(0.0,0.0,0.0);
    Eigen::Vector3f point_normal(point.normal_x,point.normal_y,point.normal_z);
    if(clusters.empty())
    {
        clusters.push_back({point, point});
        return;
    }
    else
    {
        for(int i=0; i<clusters.size(); i++)
        {          
            average_normal = Eigen::Vector3f(clusters[i][0].normal_x,clusters[i][0].normal_y,clusters[i][0].normal_z);
            TNormalize3(average_normal);
            
            if(TAngle3Normalized(point_normal, average_normal)<threshold)
            {
                clusters[i].push_back(point);
                
                clusters[i][0].x += point.x;
                clusters[i][0].y += point.y;
                clusters[i][0].z += point.z;
                
                clusters[i][0].normal_x += point.normal_x;
                clusters[i][0].normal_y += point.normal_y;
                clusters[i][0].normal_z += point.normal_z;
    
                return;
            }
        }
        clusters.push_back({point,point});
        return;
    }
} 


void group(std::vector<pcl::PointNormal>& groups, pcl::PointCloud<pcl::PointNormal>::Ptr PC, std::vector<int>& points, const double threshold)
{
    std::vector<std::vector<pcl::PointNormal>> clusters;
    clusters.reserve(points.size());
    groups.clear(); 
    Eigen::Matrix<float,3,1> average_normal(0.0,0.0,0.0);
    
    for(int index:points)
        addToClusters(clusters, PC->points[index], threshold);
    
    int cluster_num = clusters.size();
    groups.resize(cluster_num);
    
    int n = 0;
    for(int i=0; i<cluster_num; i++)
    {       
        n = clusters[i].size();
        average_normal = Eigen::Vector3f(clusters[i][0].normal_x,clusters[i][0].normal_y,clusters[i][0].normal_z);
        TNormalize3(average_normal);
        
        groups[i].x = clusters[i][0].x/(n-1);
        groups[i].y = clusters[i][0].y/(n-1);
        groups[i].z = clusters[i][0].z/(n-1);
        groups[i].normal_x = average_normal[0];
        groups[i].normal_y = average_normal[1];
        groups[i].normal_z = average_normal[2];
    }
}


void average(std::vector<pcl::PointNormal>& new_pc, std::vector<int>& points, std::vector<pcl::PointNormal>& sub_pc, const double threshold, const int num_threshold)
{
    std::vector<std::vector<pcl::PointNormal>> clusters;
    clusters.reserve(points.size());
    float average_x=0.0, average_y=0.0, average_z=0.0;
    float average_nx=0.0, average_ny=0.0, average_nz=0.0;
    for(int index:points)
    {
        addToClusters(clusters,new_pc[index],threshold);
    }
    
    sub_pc.clear();
    sub_pc.reserve(points.size());
    int irrelevant=0;
    for(int i=0; i<clusters.size(); i++)
    {
        if(clusters[i].size()-1>num_threshold)
        {
            average_x += clusters[i][0].x;
            average_y += clusters[i][0].y;
            average_z += clusters[i][0].z;

            average_nx += clusters[i][0].normal_x;
            average_ny += clusters[i][0].normal_y;
            average_nz += clusters[i][0].normal_z;
            
            irrelevant += clusters[i].size()-1;
            
        }
        else
        {
            for(int j=1; j<clusters[i].size(); j++)
            {
                sub_pc.push_back(clusters[i][j]);
            }
        }
    }
    if(irrelevant==0)
        return;
    
    pcl::PointNormal new_point;
    new_point.x = average_x/irrelevant;
    new_point.y = average_y/irrelevant;
    new_point.z = average_z/irrelevant;
    Eigen::Vector3f average_normal(average_nx,average_ny,average_nz);
    TNormalize3(average_normal);
    new_point.normal_x = average_normal[0];
    new_point.normal_y = average_normal[1];
    new_point.normal_z = average_normal[2];
    sub_pc.push_back(new_point);
}


void downSampling(pcl::PointCloud<pcl::PointNormal>::Ptr PC, pcl::PointCloud<pcl::PointNormal>::Ptr SampledPC, Eigen::Vector2f& xrange, Eigen::Vector2f& yrange, Eigen::Vector2f& zrange, const float sample_step_relative, const double threshold,const int num_threshold)
{
    std::vector<std::vector<int>> map;
    std::vector<std::vector<int>> new_map;
    std::vector<pcl::PointNormal> groups;
    std::vector<pcl::PointNormal> new_pc;
    std::vector<pcl::PointNormal> sub_pc;
    SampledPC->points.reserve(PC->points.size());
    
    int numSamplesDim = (int)(1.0/sample_step_relative);
    
    float x_r = xrange[1]-xrange[0];
    float y_r = yrange[1]-yrange[0];
    float z_r = zrange[1]-zrange[0];
    
    map.resize((numSamplesDim+1)*(numSamplesDim+1)*(numSamplesDim+1));
    
    for(int i=0; i<PC->points.size();)
    {
        int xCell = (int)((float)numSamplesDim*(PC->points[i].x-xrange[0])/x_r);
        int yCell = (int)((float)numSamplesDim*(PC->points[i].y-yrange[0])/y_r);
        int zCell = (int)((float)numSamplesDim*(PC->points[i].z-zrange[0])/z_r);
        int index = xCell*numSamplesDim*numSamplesDim+yCell*numSamplesDim+zCell;
        map[index].push_back(i);
        i+=2;
    }
    
    for(int i=0; i<map.size(); i++)
    {
        if(!map[i].empty())
        {
            group(groups,PC,map[i],threshold);
            new_pc.insert(new_pc.end(),groups.begin(),groups.end());
        }
    }
    
    int ND = (int)(1.0/(sample_step_relative*2.0));
    new_map.resize((ND+1)*(ND+1)*(ND+1));
    
    for(int i=0; i<new_pc.size();i++)
    {
        int xCell = (int)((float)ND*(new_pc[i].x-xrange[0])/x_r);
        int yCell = (int)((float)ND*(new_pc[i].y-yrange[0])/y_r);
        int zCell = (int)((float)ND*(new_pc[i].z-zrange[0])/z_r);
        int index = xCell*ND*ND+yCell*ND+zCell;
        new_map[index].push_back(i);
    }
    
    
    for(int i=0; i<new_map.size(); i++)
    {
        if(!new_map[i].empty())
        {
            average(new_pc,new_map[i],sub_pc,threshold,num_threshold);
            for(int j=0; j<sub_pc.size(); j++)
            {
                SampledPC->push_back(sub_pc[j]);
            }
        }
    }
    
}

//simple down-sampling
void downSampling(pcl::PointCloud<pcl::PointNormal>::Ptr PC, pcl::PointCloud<pcl::PointNormal>::Ptr SampledPC, const float leaf_size)
{
    pcl::VoxelGrid<pcl::PointNormal> vox;
    vox.setInputCloud(PC);
    vox.setLeafSize(leaf_size,leaf_size,leaf_size);
    vox.filter(*SampledPC);
}

void computeBboxStd(pcl::PointCloud<pcl::PointNormal>::Ptr PC, Eigen::Vector2f& xrange, Eigen::Vector2f& yrange, Eigen::Vector2f& zrange)
{
    pcl::PointNormal min, max;
    pcl::getMinMax3D(*PC,min,max);
    xrange[0] = min.x;
    xrange[1] = max.x;
    yrange[0] = min.y;
    yrange[1] = max.y;
    zrange[0] = min.z;
    zrange[1] = max.z;
}

//compute normals using PCA
void computeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr PC, pcl::PointCloud<pcl::PointNormal>::Ptr PCNormals, bool FlipViewpoint)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr OutlierRemovedPC(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr SampledPC(new pcl::PointCloud<pcl::PointXYZ>);

    
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    outrem.setInputCloud(PC);
    outrem.setRadiusSearch(3.0);
    outrem.setMinNeighborsInRadius(3);
    outrem.filter(*OutlierRemovedPC);
    
    //OutlierRemovedPC = PC;

    pcl::VoxelGrid<pcl::PointXYZ> vox;
    vox.setInputCloud(OutlierRemovedPC);
    vox.setLeafSize(0.5f,0.5f,0.5f);
    vox.filter(*SampledPC);
    
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    //normalEstimation.setIndices()
    normalEstimation.setKSearch(15);
    normalEstimation.setInputCloud(SampledPC);
    if(FlipViewpoint)
        normalEstimation.setViewPoint(0.0,0.0,0.0);
    
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);
    pcl::concatenateFields(*SampledPC,*normals,*PCNormals);
    
    std::vector<int> mapping;
    pcl::removeNaNFromPointCloud(*PCNormals,*PCNormals,mapping);

}

void computeNormals_on(pcl::PointCloud<pcl::PointXYZ>::Ptr PC, pcl::PointCloud<pcl::PointNormal>::Ptr PCNormals)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr pcn(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    
    pcl::IntegralImageNormalEstimation<pcl::PointXYZ,pcl::Normal> ne;
    ne.setInputCloud(PC);
    
    ne.setNormalEstimationMethod(ne.AVERAGE_DEPTH_CHANGE);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    
    ne.compute(*normals);
    pcl::concatenateFields(*PC,*normals,*pcn);
    
    bool pInf=false,pNan=false,nInf=false,nNan=false;
    //pcl::PointNormal point;
    //Eigen::Vector3d vp(0.0,0.0,0.0);
    for(int i=0; i<pcn->points.size(); i++)
    {
        pInf = false;
        pNan = false;
        nInf = false;
        nNan = false;
        
        if(isinf(pcn->points[i].x) || isinf(pcn->points[i].y) || isinf(pcn->points[i].z))
            pInf = true;
        if(isnan(pcn->points[i].x) || isnan(pcn->points[i].y) || isnan(pcn->points[i].z))
            pNan = true;
        if(isinf(pcn->points[i].normal[0]) || isinf(pcn->points[i].normal[1]) || isinf(pcn->points[i].normal[2]))
            nInf = true;
        if(isnan(pcn->points[i].normal[0]) || isnan(pcn->points[i].normal[1]) || isnan(pcn->points[i].normal[2]))
            nNan = true;
        if(!pInf && !pNan && !nInf && !nNan)
        {
            //PCNormals->push_back(pcn->points[i]);
            pcn->points[i].x = 1000*pcn->points[i].x;
            pcn->points[i].y = 1000*pcn->points[i].y;
            pcn->points[i].z = 1000*pcn->points[i].z;
            //flipNormalViewpoint(vp,pcn->points[i]);
            PCNormals->push_back(pcn->points[i]);
        }
    }
}

void removeBG(pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene, pcl::PointCloud<pcl::PointXYZRGB>::Ptr empty_scene, pcl::PointCloud<pcl::PointXYZ>::Ptr object, const double positionThreshold, const double colorThreshold)
{
    
    int width = scene->width;
    int height = scene->height;
    
    object->width = width;
    object->height = height;
    object->resize(width*height);
    object->is_dense = false;
    
    for(int i=0; i<width; i++)
    {
        for(int j=0; j<height; j++)
        {
            bool flag = false;
            
            if(isnan(scene->at(i,j).z))
                continue;
            if(corr(scene->at(i,j),empty_scene->at(i,j),positionThreshold,colorThreshold))
                flag = true;
          
            if(!flag)
            {
                object->at(i,j).x = scene->at(i,j).x;
                object->at(i,j).y = scene->at(i,j).y;
                object->at(i,j).z = scene->at(i,j).z;
                
            }
            else
            {
                object->at(i,j).x = std::numeric_limits<float>::quiet_NaN();
                object->at(i,j).y = std::numeric_limits<float>::quiet_NaN();
                object->at(i,j).z = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void removeBG(pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene, pcl::PointCloud<pcl::PointXYZRGB>::Ptr empty_scene, pcl::PointCloud<pcl::PointXYZ>::Ptr object, const double positionThreshold)
{
    int width = scene->width;
    int height = scene->height;
    
    object->width = width;
    object->height = height;
    object->resize(width*height);
    object->is_dense = false;
    
    for(int i=0; i<width; i++)
    {
        for(int j=0; j<height; j++)
        {
            bool flag = false;
            
            if(isnan(scene->at(i,j).z))
                continue;
            if(corr(scene->at(i,j),empty_scene->at(i,j),positionThreshold))
                flag = true;
          
            if(!flag)
            {
                object->at(i,j).x = scene->at(i,j).x;
                object->at(i,j).y = scene->at(i,j).y;
                object->at(i,j).z = scene->at(i,j).z;
                
            }
            else
            {
                object->at(i,j).x = std::numeric_limits<float>::quiet_NaN();
                object->at(i,j).y = std::numeric_limits<float>::quiet_NaN();
                object->at(i,j).z = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}


void removeBGWithPatch(pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene, pcl::PointCloud<pcl::PointXYZRGB>::Ptr empty_scene, pcl::PointCloud<pcl::PointXYZ>::Ptr object, const double positionThreshold, const double colorThreshold, const int patch_size)
{
    //pcl::PointCloud<pcl::PointXYZ>::Ptr leftPoints(new pcl::PointCloud<pcl::PointXYZ>);
    int width = scene->width;
    int height = scene->height;
    
    object->width = width;
    object->height = height;
    object->resize(width*height);
    object->is_dense = false;
    
    for(int i=0; i<width; i++)
    {
        for(int j=0; j<height; j++)
        {
            bool flag = false;
            
            if(isnan(scene->at(i,j).z))
            {
                object->at(i,j).x = object->at(i,j).y = object->at(i,j).z = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            if(!isnan(empty_scene->at(i,j).z) && corr(scene->at(i,j),empty_scene->at(i,j),positionThreshold,colorThreshold))
                flag = true;
            if(!flag && searchInPatch(scene,empty_scene,i,j,patch_size,positionThreshold,colorThreshold))
                flag = true;
            if(!flag)
            {
                object->at(i,j).x = scene->at(i,j).x/1000.0;
                object->at(i,j).y = scene->at(i,j).y/1000.0;
                object->at(i,j).z = scene->at(i,j).z/1000.0;
            }
            else
                object->at(i,j).x = object->at(i,j).y = object->at(i,j).z = std::numeric_limits<float>::quiet_NaN();
        }
    }
}

bool searchInPatch(pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene, pcl::PointCloud<pcl::PointXYZRGB>::Ptr empty_scene, int i, int j, const int patch_size, const double positionThreshold, const double colorThreshold)
{
    int left_bound=0, right_bound=0, up_bound=0, down_bound=0;
    int width=scene->width, height=scene->height;
    
    if((i-patch_size/2) < 0)
        left_bound = 0;
    else
        left_bound = i-patch_size/2;
    
    if((i+patch_size/2) > (width-1))
        right_bound = width-1;
    else
        right_bound = i+patch_size/2;
    
    if((j-patch_size/2) < 0)
        up_bound = 0;
    else
        up_bound = j-patch_size/2;
    
    if((j+patch_size/2) > (height-1))
        down_bound = height-1;
    else
        down_bound = j+patch_size/2;
    
   
    for(int w=left_bound; w<=right_bound; w++)
        for(int h=up_bound; h<=down_bound; h++)
        {
            if(!isnan(empty_scene->at(w,h).z) && corr(scene->at(i,j),empty_scene->at(w,h),positionThreshold,colorThreshold))
                return true;
        }
        
    return false;
}


//compute the color difference in the LAB space, see https://www.compuphase.com/cmetric.htm
bool corr(pcl::PointXYZRGB p1, pcl::PointXYZRGB p2, const double positionThreshold, const double colorThreshold)
{
    Eigen::Vector3f pos(p1.x-p2.x,p1.y-p2.y,p1.z-p2.z);
    double posDiff = pos.norm();
    int r1 = p1.r, g1 = p1.g, b1 = p1.b;
    int r2 = p2.r, g2 = p2.g, b2 = p2.b;
    double rmean = (r1+r2)/2.0;
    double dr = r1-r2;
    double dg = g1-g2;
    double db = b1-b2;
    double colorDiff = (double)sqrt((2.0+rmean/256.0)*dr*dr+4.0*dg*dg+(2.0+(255.0-rmean)/256.0)*db*db);
    if(posDiff<positionThreshold && colorDiff<colorThreshold)
        return true;
    
    return false;
}

bool corr(pcl::PointXYZRGB p1, pcl::PointXYZRGB p2, const double positionThreshold)
{
    Eigen::Vector3f pos(p1.x-p2.x,p1.y-p2.y,p1.z-p2.z);
    double posDiff = pos.norm();

    if(posDiff<positionThreshold)
        return true;
    
    return false;
}


void transformPointCloud(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, Eigen::Matrix4d& T)
{
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.topRightCorner<3,1>();
    
    for(int i=0; i<cloud->points.size(); i++)
    {
        pcl::PointNormal p = cloud->points[i];
        
        Eigen::Vector3d pos(p.x,p.y,p.z);
        Eigen::Vector3d nor(p.x+p.normal_x,p.y+p.normal_y,p.z+p.normal_z);
        
        Eigen::Vector3d new_pos = R*pos+t;
        Eigen::Vector3d new_nor = R*nor+t;
        
        cloud->points[i].x = new_pos(0);
        cloud->points[i].y = new_pos(1);
        cloud->points[i].z = new_pos(2);
        
        Eigen::Vector3d n = new_nor-new_pos;
        n.normalize();
        cloud->points[i].normal_x = n(0);
        cloud->points[i].normal_y = n(1);
        cloud->points[i].normal_z = n(2);
    }
}

}
