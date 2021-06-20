#include "../include/myppf/headers.hpp"
#include "../include/myppf/hash_murmur64.hpp"

namespace myppf
{

//static const size_t PPF_LENGTH = 5;
  
static bool sortPoses(const Pose3DPtr& a, const Pose3DPtr& b)
{
  CV_Assert(!a.empty() && !b.empty());
  return (a->numVotes > b->numVotes);
}
  
static bool sortPoseClusters(const PoseCluster3DPtr& a, const PoseCluster3DPtr& b)
{
  CV_Assert(!a.empty() && !b.empty());
  return (a->numVotes > b->numVotes);
}
  
static KeyType hashPPF(const Eigen::Vector4f& f, const double AngleStep, const double DistanceStep)
{
  Eigen::Vector4i key((int)(f[0]/AngleStep), (int)(f[1]/AngleStep), (int)(f[2]/AngleStep), (int)(f[3]/DistanceStep));
  KeyType hashKey[2] = {0,0};
  
  hashMurmurx64(key.data(),4*sizeof(int),42,&hashKey[0]);
  return hashKey[0];
}

static KeyType hashPPF(const Eigen::Vector4i& key)
{
  KeyType hashKey[2] = {0,0};
  hashMurmurx64(key.data(),4*sizeof(int),42,&hashKey[0]);
  return hashKey[0];
}
  
static double computeAlpha(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1, const Eigen::Vector3f& p2)
{
  Eigen::Vector3d Tmg, mpt;
  Eigen::Matrix3d R;
  double alpha;
  
  computeTransformRT(p1,n1,R,Tmg);
  mpt = Tmg + R*p2.cast<double>();
  alpha = atan2(-mpt[2], mpt[1]);
  
  /*if(alpha != alpha)
    return 0;
  
  if(sin(alpha)*mpt[2]<0.0)
    alpha = -alpha;
  
  return (-alpha);*/
  
  if(sin(alpha)*mpt[2]>0.0)
    alpha = -alpha;
  return alpha;
}

PPFDetector::PPFDetector()
{
  sampling_step_relative = 0.01;
  distance_step_relative = 0.05;
  scene_sample_step = (int)(1/0.04);
  angle_step_relative = 30;
  angle_step_radians = (360.0/angle_step_relative)*M_PI/180.0;
  angle_step = angle_step_radians;
  trained = false;

  hash_table.clear();
  hash_nodes = NULL;

  //setClusterParams(10.0,1.5,true);
  setClusterParams(0.5,0.1,true);
}

void PPFDetector::setClusterParams(const double positionThreshold, const double rotationThreshold, const bool useWeightedClustering)
{
  if (positionThreshold<0)
    position_threshold = sampling_step_relative;
  else
    position_threshold = positionThreshold;

  if (rotationThreshold<0)
    rotation_threshold = ((360/angle_step) / 180.0 * M_PI);
  else
    rotation_threshold = rotationThreshold;

  use_weighted_avg = useWeightedClustering;
}

void PPFDetector::clearTrainedModel()
{
  if(this->hash_nodes)
  {
    free(this->hash_nodes);
    this->hash_nodes = 0;
  }
  
  if(!hash_table.empty())
  {
    hash_table.clear();
  }
}

//compute the PPF
void PPFDetector::computePPFFeatures(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1, const Eigen::Vector3f& p2, const Eigen::Vector3f& n2, Eigen::Vector4f& f)
{
  Eigen::Vector3f d(p2-p1);
  f[3] = d.norm();
  if (f[3] <= EPS)
    return;
  d /= f[3];

  f[0] = TAngle3Normalized(n1, d);
  f[1] = TAngle3Normalized(n2, d);
  f[2] = TAngle3Normalized(n1, n2);
}

Eigen::Vector4i PPFDetector::neighborInd(const Eigen::Vector4f& f, const double AngleStep, const double DistanceStep)
{
  Eigen::Vector4i ind;
  Eigen::Vector4f eq;
  eq[0] = (float)(f[0]/DistanceStep-floor(f[0]/DistanceStep));
  eq[1] = (float)(f[1]/AngleStep-floor(f[1]/AngleStep));
  eq[2] = (float)(f[2]/AngleStep-floor(f[2]/AngleStep));
  eq[3] = (float)(f[3]/AngleStep-floor(f[3]/AngleStep));
  for(int i=0; i<4; i++)
  {
    if(eq[i]<(float)(1.0/3.0))
      ind[i] = -1;
    else if(eq[i]>(float)(2.0/3.0))
      ind[i] = 1;
    else 
      ind[i] = 0;
  }
  return ind;
}
  
std::vector<KeyType> PPFDetector::searchNeighbors(const Eigen::Vector4f& f, const double AngleStep, const double DistanceStep)
{
  std::vector<KeyType> neighbors;
  Eigen::Vector4i ind = neighborInd(f,AngleStep,DistanceStep);
  std::vector<std::vector<int>> vec;
  for(int i=0; i<4; i++)
  {
    if(ind[i] == 0)
      vec.push_back({0});
    else
      vec.push_back({0,ind[i]});
  }
  for(int v0:vec[0])
  {
    for(int v1:vec[1])
    {
      for(int v2:vec[2])
      {
        for(int v3:vec[3])
        {
          Eigen::Vector4i key((int)(f[0]/AngleStep)+v0, (int)(f[1]/DistanceStep)+v1, (int)(f[2]/DistanceStep)+v2, (int)(f[3]/DistanceStep)+v3);
          KeyType hashValue = hashPPF(key);
          neighbors.push_back(hashValue);
        }
      }
    }
  }
  return neighbors;
}

  
PPFDetector::~PPFDetector()
{
  clearTrainedModel();
}

void PPFDetector::trainModel(pcl::PointCloud<pcl::PointNormal>::Ptr model)
{
  pcl::PointCloud<pcl::PointNormal>::Ptr sampled(new pcl::PointCloud<pcl::PointNormal>);
  Eigen::Vector2f xr,yr,zr;
  computeBboxStd(model,xr,yr,zr);
  
  float dx = xr[1]-xr[0];
  float dy = yr[1]-yr[0];
  float dz = zr[1]-zr[0];
  float diameter = sqrt(dx*dx+dy*dy+dz*dz);
  
  this->model_diameter = diameter;
  
  float distanceStep = (float)(diameter*sampling_step_relative);
  //downSampling(model,sampled,xr,yr,zr,sampling_step_relative*5,0.05,5);
  downSampling(model,sampled,7.5);
  
  int num_points = sampled->points.size();
  int num_ref = num_points;
  int size = num_ref*num_ref;
  
  
  //hashtable_int* hashTable = hashtableCreate(size,NULL);
  //ppf = cv::Mat(size,PPF_LENGTH,CV_32FC1);
  
  hash_nodes = (THash*)calloc(size,sizeof(THash));
  
  hash_table.reserve(size);
  
  double lamda = 0.98;
  for(int i=0; i<num_ref; i++)
  {
    if(i%10 == 0)
        std::cout<<"trained "<<i*100/num_ref+1<<"%"<<std::endl;
    pcl::PointNormal p1 = sampled->points[i];
    const Eigen::Vector3f pos1(p1.x,p1.y,p1.z);
    const Eigen::Vector3f nor1(p1.normal[0],p1.normal[1],p1.normal[2]);
    
    for(int j=0; j<num_points; j++)
    {
      if(j != i)
      {
        pcl::PointNormal p2 = sampled->points[j];
        const Eigen::Vector3f pos2(p2.x,p2.y,p2.z);
        const Eigen::Vector3f nor2(p2.normal[0],p2.normal[1],p2.normal[2]);
        
        Eigen::Vector4f f(0.0,0.0,0.0,0.0);
        computePPFFeatures(pos1,nor1,pos2,nor2,f);
        if(f.hasNaN())
            continue;
        
        KeyType hashValue = hashPPF(f,angle_step_radians,distanceStep);
        
        double alpha = computeAlpha(pos1,nor1,pos2);
        uint Ind = i*num_points+j;
        
        double dp = nor1[0]*nor2[0]+nor1[1]*nor2[1]+nor1[2]*nor2[2];
        double voteVal = 1-lamda*abs(dp);
        
        THash* hashNode = &hash_nodes[Ind];
        hashNode->i = i;
        hashNode->ppfInd = Ind;
        hashNode->alpha = alpha;
        hashNode->voteVal = voteVal;
        
        //hashtableInsertHashed(hashTable, hashValue, (void*)hashNode);
        if(hash_table.find(hashValue)!= hash_table.end())
            hash_table[hashValue].push_back(hashNode);
        else
            hash_table.emplace(hashValue,std::vector<THash*>{hashNode});
        
        //ppf.ptr<float>(Ind)[0] = f[0];
        //ppf.ptr<float>(Ind)[1] = f[1];
        //ppf.ptr<float>(Ind)[2] = f[2];
        //ppf.ptr<float>(Ind)[3] = f[3];
        //ppf.ptr<float>(Ind)[4] = (float)(alpha);
      }
    }
  }
  
  angle_step = angle_step_radians;
  distance_step = distanceStep;
  //hash_table = hashTable;
  num_ref_points = num_ref;
  sampled_pc = sampled;
  trained = true;
}


void PPFDetector::setScene(const char* filename,const float relativeSceneDistance, const double threshold, const int num_threshold)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr empty_scene(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr object(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal>::Ptr PCNormals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr SampledPC(new pcl::PointCloud<pcl::PointNormal>);
    
    loadPLYRGB(scene, filename);
    loadPLYRGB(empty_scene, "/home/yongqi/project/data/scene/empty_scene.ply");
    removeBG(scene,empty_scene,object,0.001);
    //removeBGWithPatch(scene,empty_scene,object,0.002,150,5);
    
    computeNormals_on(object,PCNormals);
    
    /*pcl::PointCloud<pcl::PointNormal>::Ptr model(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr sampledmodel(new pcl::PointCloud<pcl::PointNormal>);
    loadOBJ(model, "/home/yongqi/project/data/scene/Cube_35mm.obj", 50000, true);
    downSampling(model,sampledmodel,1.5);
    Eigen::Matrix4d T;
    T<<1.0,  0.0,   0.0,  -100.758,
 0.0,   1.0,  0.0,  -120.963,
 0.0,  0.0, 1.0,   480.069,
        0,         0,         0,         1;   
    pcl::transformPointCloud(*sampledmodel,*sampledmodel,T);
    for(int i=0; i<sampledmodel->points.size(); i++)
        PCNormals->points.push_back(sampledmodel->points[i]);*/
    
    Eigen::Vector2f xr,yr,zr;
    computeBboxStd(PCNormals,xr,yr,zr);
    //downSampling(PCNormals,SampledPC,xr,yr,zr,relativeSceneDistance,threshold,num_threshold);
    downSampling(PCNormals,SampledPC,2.0f);
    
    sampled_scene = SampledPC;
    
    
    /*boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normals"));
    //viewer->addPointCloud<pcl::PointXYZ>(object,"cloud");
    //viewer->setBackgroundColor(0.0,0.0,0.0);
    //viewer->addPointCloudNormals<pcl::PointNormal,pcl::PointNormal>(SampledPC,SampledPC,1,2.0);
    viewer->addPointCloudNormals<pcl::PointNormal,pcl::PointNormal>(SampledPC,SampledPC,1,2);
    
    while(!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
    */
}


//matching
bool PPFDetector::matchPose(const Pose3D& sourcePose, const Pose3D& targetPose)
{
  Eigen::Vector3d v = targetPose.t-sourcePose.t;
  double dNorm = v.norm();
  const double phi = fabs(targetPose.angle-sourcePose.angle);
  return (dNorm<this->position_threshold && phi<this->rotation_threshold);
}


void PPFDetector::clusterPoses(std::vector<Pose3DPtr>& poseList, int numPoses, std::vector<Pose3DPtr>& finalPoses)
{
  std::vector<PoseCluster3DPtr> poseClusters;
  
  finalPoses.clear();

  // sort the poses for stability
  std::sort(poseList.begin(), poseList.end(), sortPoses);

  for (int i=0; i<numPoses; i++)
  {
    Pose3DPtr pose = poseList[i];
    bool assigned = false;

    // search all clusters
    for (size_t j=0; j<poseClusters.size() && !assigned; j++)
    {
      const Pose3DPtr poseCenter = poseClusters[j]->poseList[0];
      if (matchPose(*pose, *poseCenter))
      {
        poseClusters[j]->addPose(pose);
        assigned = true;
      }
    }

    if (!assigned)
    {
      poseClusters.push_back(PoseCluster3DPtr(new PoseCluster3D(pose)));
    }
  }
  
  //sort the clusters
  std::sort(poseClusters.begin(),poseClusters.end(),sortPoseClusters);
  finalPoses.resize(poseClusters.size());
  
  if (use_weighted_avg)
  {
#if defined _OPENMP
#pragma omp parallel for
#endif
    // uses weighting by the number of votes
    for (int i=0; i<static_cast<int>(poseClusters.size()); i++)
    {
      // We could only average the quaternions
      Eigen::Vector4d qAvg = Eigen::Vector4d::Zero();
      Eigen::Vector3d tAvg = Eigen::Vector3d::Zero();

      // Perform the final averaging
      PoseCluster3DPtr curCluster = poseClusters[i];
      std::vector<Pose3DPtr> curPoses = curCluster->poseList;
      int curSize = (int)curPoses.size();
      double numTotalVotes = 0.0;

      for (int j=0; j<curSize; j++)
        numTotalVotes += curPoses[j]->numVotes;

      double wSum=0;

      for (int j=0; j<curSize; j++)
      {
        const double w = (double)curPoses[j]->numVotes / (double)numTotalVotes;

        qAvg += w * curPoses[j]->q.coeffs();
        tAvg += w * curPoses[j]->t;
        wSum += w;
      }

      tAvg *= 1.0 / wSum;
      qAvg *= 1.0 / wSum;

      Eigen::Quaterniond qua(qAvg[3],qAvg[0],qAvg[1],qAvg[2]);
      qua.normalize();
      
      curPoses[0]->updatePoseQuat(qua, tAvg);
      curPoses[0]->numVotes=curCluster->numVotes;

      finalPoses[i]=curPoses[0]->clone();
    }
  }
  else
  {
#if defined _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<static_cast<int>(poseClusters.size()); i++)
    {
      // We could only average the quaternions
      Eigen::Vector4d qAvg = Eigen::Vector4d::Zero();
      Eigen::Vector3d tAvg = Eigen::Vector3d::Zero();

      // Perform the final averaging
      PoseCluster3DPtr curCluster = poseClusters[i];
      std::vector<Pose3DPtr> curPoses = curCluster->poseList;
      const int curSize = (int)curPoses.size();

      for (int j=0; j<curSize; j++)
      {
        qAvg += curPoses[j]->q.coeffs();
        tAvg += curPoses[j]->t;
      }

      tAvg *= 1.0 / curSize;
      qAvg *= 1.0 / curSize;

      Eigen::Quaterniond qua(qAvg[3],qAvg[0],qAvg[1],qAvg[2]);
      qua.normalize();
      
      curPoses[0]->updatePoseQuat(qua, tAvg);
      curPoses[0]->numVotes=curCluster->numVotes;

      finalPoses[i]=curPoses[0]->clone();
    }
  }
  poseClusters.clear();
}
  

void PPFDetector::save_model(const char* file)
{
    std::ofstream of(file);
  
    of<<"model parameters"<<endl;
    of<<"model diameter "<<model_diameter<<endl;
    of<<"hash table size "<<hash_table.size()<<endl;
    of<<"point cloud size "<<sampled_pc->points.size()<<endl;
    //of<<"ppf rows "<<ppf.rows<<endl;
    of<<"end header"<<endl;
    
    for(const auto &item:hash_table)
    {
        of<<item.first<<" "<<item.second.size()<<" ";
        for(const auto node:item.second)
        {
            //of<<node->id<<" "<<node->i<<" "<<node->ppfInd<<" ";
            of<<node->ppfInd<<" "<<node->i<<" "<<node->alpha<<" "<<node->voteVal<<" ";
        }
        of<<endl;
    }
    
    for(int i=0; i<sampled_pc->points.size(); i++)
    {
        of<<sampled_pc->points[i].x<<" "<<sampled_pc->points[i].y<<" "<<sampled_pc->points[i].z<<" "<<sampled_pc->points[i].normal_x<<" "<<sampled_pc->points[i].normal_y<<" "<<sampled_pc->points[i].normal_z<<endl;
    }
    
    /*for(int i=0; i<ppf.rows; i++)
    {
        float* data = ppf.ptr<float>(i);
        of<<data[0]<<" "<<data[1]<<" "<<data[2]<<" "<<data[3]<<" "<<data[4]<<endl;
    }*/
    
    of.close();
}

void PPFDetector::load_model(const char* file)
{
    std::ifstream in(file);
    
    std::string tbSZ = "hash table size ";
    std::string pcSZ = "point cloud size ";
    std::string diameter_flag = "model diameter ";
    //std::string ppf_rows_flag = "ppf rows ";
    int p1 = diameter_flag.size();
    int p2 = tbSZ.size();
    int p3 = pcSZ.size();
    //int p4 = ppf_rows_flag.size();
    
    int n1,n2,ppf_rows;
    
    std::string line;
    while(!in.eof())
    {
        getline(in,line);
        if(line.find(diameter_flag)!=std::string::npos)
        {
            std::string num = line.substr(p1);
            model_diameter = atof(num.c_str());
        }
        if(line.find(tbSZ)!=std::string::npos)
        {
            std::string num = line.substr(p2);
            n1 = atoi(num.c_str());
        }
        if(line.find(pcSZ)!= std::string::npos)
        {
            std::string num = line.substr(p3);
            n2 = atoi(num.c_str());
        }
        /*if(line.find(ppf_rows_flag)!=std::string::npos)
        {
            std::string num = line.substr(p4);
            ppf_rows = atoi(num.c_str());
        }*/
           
        if(line=="end header")
        {
            break;
        }
    }
    
    std::unordered_map<KeyType,std::vector<THash*>> table;
    table.reserve(n1);
    std::vector<THash*> value;
    
    KeyType key;
    int nums,i,ppfInd;
    double alpha,voteVal;
    
    while(!in.eof() && n1>0)
    {
        in>>key>>nums;
        //cout<<key<<" "<<nums<<endl;;
        value.clear();
        while(nums>0)
        {
            //in>>id>>i>>ppfInd;
            in>>ppfInd>>i>>alpha>>voteVal;
            
            THash* node(new THash);
            
            node->ppfInd = ppfInd;
            node->i = i;
            node->alpha = alpha;
            node->voteVal = voteVal;
            value.push_back(node);
            nums--;
        }
        
        table.emplace(key,value);
        getline(in,line);
        n1--;
    }
    
    hash_table = table;
    
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>);
    cloud->resize(n2);
    float x,y,z,nx,ny,nz;
    int n=0;
    while(!in.eof() && n<n2)
    {
        in>>x>>y>>z>>nx>>ny>>nz;
        //cout<<x<<" "<<y<<" "<<z<<" "<<nx<<" "<<ny<<" "<<nz<<endl;
        if(in.fail())
            break;
        cloud->points[n].x = x;
        cloud->points[n].y = y;
        cloud->points[n].z = z;
        cloud->points[n].normal_x = nx;
        cloud->points[n].normal_y = ny;
        cloud->points[n].normal_z = nz;
        
        getline(in,line);
        n++;
    }
    
    sampled_pc = cloud;
    
    /*cv::Mat saved_ppf(ppf_rows,PPF_LENGTH,CV_32FC1);
    n = 0;
    float f1,f2,f3,f4,alpha;
    while(!in.eof() && n<ppf_rows)
    {
        in>>f1>>f2>>f3>>f4>>alpha;
        if(in.fail())
            break;
        float* data = saved_ppf.ptr<float>(n);
        data[0] = f1;
        data[1] = f2;
        data[2] = f3;
        data[3] = f4;
        data[4] = alpha;
    }
    
    ppf = saved_ppf;*/
    
    trained = true;
    
    in.close();
    
}

void PPFDetector::save_poses(const char* file)
{
    std::ofstream of(file);
    
    /*for(const auto &pose:poses)
    {
        of<<pose->pose.matrix()<<endl;
    }*/
    
    for(int i=0; i<poses.size(); i++)
    {
        of<<poses[i]->pose.matrix()<<endl;
    }
    
    of.close();
}

void PPFDetector::load_poses(const char* file,std::vector<Eigen::Matrix4d>& pose_list)
{
    std::ifstream in(file);
    
    std::vector<double> vec;
    double n1,n2,n3,n4;
    std::string line;
    
    Eigen::Matrix4d pose;
    int i=0;
    while(!in.eof())
    {
        in>>n1>>n2>>n3>>n4;
        pose.row(i)<<n1,n2,n3,n4;
        i++;
        getline(in,line);
        if(i>3)
        {
            i=0;
            pose_list.push_back(pose);
        }
    }
    
    in.close();
}



void PPFDetector::match(std::vector<Pose3DPtr>& results, const double relativeSampleStep, const double relativeSceneDistance, const int minimum_supports)
{
  if(!trained)
      std::cerr<<"the model haven't been trained"<<endl;
    //CV_ERROR("the model haven't been trained");
  CV_Assert(relativeSampleStep>0.0 && relativeSampleStep<1.0);
  
  scene_sample_step = (int)(1.0/relativeSampleStep);
  distance_step = model_diameter*sampling_step_relative;
  
  int numAngles = (int) (floor (2 * M_PI / angle_step));
  float distanceStep = (float)distance_step;
  //uint n = num_ref_points;
  uint n = sampled_pc->points.size();
  std::vector<Pose3DPtr> poseList;
  int sceneSamplingStep = scene_sample_step;
  
  //compute bbox
  //Eigen::Vector2f xr,yr,zr;
  //computeBboxStd(scene,xr,yr,zr);
  
  pcl::PointCloud<pcl::PointNormal>::Ptr scene_sampled(new pcl::PointCloud<pcl::PointNormal>);
  scene_sampled = this->sampled_scene;
  poseList.reserve(scene_sampled->points.size()/sceneSamplingStep+4);
  
               
  pcl::KdTreeFLANN<pcl::PointNormal> kdtree;
  kdtree.setInputCloud(scene_sampled);
  std::vector<int> indices;
  std::vector<float> distances;

#if defined _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<scene_sampled->points.size(); i += sceneSamplingStep)
  {
    std::cout<<"matching "<<i*100/scene_sampled->points.size()<<"%"<<std::endl;
    uint refIndMax = 0, alphaIndMax = 0;
    double maxVotes = 0;
    pcl::PointNormal pt1 = scene_sampled->points[i];

    const Eigen::Vector3f p1(pt1.x,pt1.y,pt1.z);
    const Eigen::Vector3f n1(pt1.normal[0],pt1.normal[1],pt1.normal[2]);
    Eigen::Matrix3d Rsg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d RInv = Eigen::Matrix3d::Zero();
    Eigen::Vector3d tsg(0.0,0.0,0.0);

    computeTransformRT(p1,n1,Rsg,tsg);
    
    double* accumulator = (double*)calloc(numAngles*n,sizeof(double));

    indices.clear();
    distances.clear();
    if(kdtree.radiusSearch(pt1,1.2*model_diameter,indices,distances)<0)
     continue;

    for(int j=0; j<indices.size(); j++)
    {
      int index = indices[j];
      if(index!=i)
      {
        pcl::PointNormal pt2 = scene_sampled->points[index];
        const Eigen::Vector3f p2(pt2.x,pt2.y,pt2.z);
        const Eigen::Vector3f n2(pt2.normal[0],pt2.normal[1],pt2.normal[2]);

        Eigen::Vector3d p2t;
        double alpha_scene;

        Eigen::Vector4f f = Eigen::Vector4f::Zero();
        computePPFFeatures(p1,n1,p2,n2,f);

        std::vector<KeyType> hashValues = searchNeighbors(f,angle_step,distanceStep);

        p2t = tsg+Rsg*p2.cast<double>();
        alpha_scene=atan2(-p2t[2], p2t[1]);
        /*if ( alpha_scene != alpha_scene)
        {
          continue;
        }
        if(sin(alpha_scene)*p2t[2]<0.0)
          alpha_scene=-alpha_scene;
        alpha_scene=-alpha_scene;*/
        if(sin(alpha_scene)*p2t[2]>0.0)
          alpha_scene = -alpha_scene;

        for(KeyType hashValue:hashValues)
        {
          //hashnode_i* node = hashtableGetBucketHashed(hash_table,(hashValue));
          if(hash_table.find(hashValue) != hash_table.end())
          {
            std::vector<THash*> nodes = hash_table[hashValue];
            for(auto node:nodes)
            {
                int corrI = (int)node->i;
                int ppfInd = (int)node->ppfInd;
                double alpha_model = (double)node->alpha;
                double voteVal = (double)node->voteVal;
                //float* ppfCorrScene = ppf.ptr<float>(ppfInd);
                //double alpha_model = (double)ppfCorrScene[PPF_LENGTH-1];
                double alpha = alpha_model-alpha_scene;
                
                int alpha_index = (int)(numAngles*(alpha+2*M_PI)/(4*M_PI));
                uint accIndex = corrI*numAngles + alpha_index;
                
                accumulator[accIndex]++;
                //accumulator[accIndex] = accumulator[accIndex]+voteVal;
            }
          }
        }
        
      }
    }
    
    //maximize the accumulator
    for (uint k = 0; k < n; k++)
    {
      for (int j = 0; j < numAngles; j++)
      {
        const uint accInd = k*numAngles + j;
        const double accVal = accumulator[accInd];
        if (accVal > maxVotes)
        {
          maxVotes = accVal;
          refIndMax = k;
          alphaIndMax = j;
        }
#if !defined (_OPENMP)
        accumulator[accInd] = 0;
#endif
      }
    }
    
    if(maxVotes>minimum_supports)
    {
      Eigen::Vector3d tInv,tmg;
      Eigen::Matrix3d Rmg;
      RInv = Rsg.transpose();
      tInv = -RInv*tsg;

      Eigen::Isometry3d TsgInv = Eigen::Isometry3d::Identity();
      TsgInv.matrix().topRightCorner<3,1>() = tInv;
      TsgInv.matrix().block<3,3>(0,0) = RInv;
      
      pcl::PointNormal ref = sampled_pc->points[refIndMax];
      
      const Eigen::Vector3f pMax(ref.x,ref.y,ref.z);
      const Eigen::Vector3f nMax(ref.normal_x,ref.normal_y,ref.normal_z);
      
      computeTransformRT(pMax,nMax,Rmg,tmg);
      
      Eigen::Isometry3d Tmg = Eigen::Isometry3d::Identity();
      Tmg.matrix().topRightCorner<3,1>() = tmg;
      Tmg.matrix().block<3,3>(0,0) = Rmg;
      
      //convert alpha_index to alpha
      int alpha_index = alphaIndMax;
      double alpha = (alpha_index*(4*M_PI))/numAngles-2*M_PI;

      Eigen::Isometry3d Talpha = Eigen::Isometry3d::Identity();
      Eigen::Matrix3d R;
      Eigen::Vector3d t(0.0,0.0,0.0);
      getUnitXRotation(alpha,R);
      
      Talpha.matrix().topRightCorner<3,1>() = t;
      Talpha.matrix().block<3,3>(0,0) =  R;

      Eigen::Isometry3d rawPose = TsgInv*(Talpha*Tmg);
      
      Pose3DPtr pose(new Pose3D(alpha,refIndMax,maxVotes));
      pose->updatePose(rawPose);
      #if defined (_OPENMP)
      #pragma critical
      #endif
      {
        poseList.push_back(pose);
      }
    }
    free(accumulator);
  }
  
  std::cout<<"total number of poses "<<poseList.size()<<std::endl;
               
  int numPosesAdded = poseList.size();
  clusterPoses(poseList,numPosesAdded,results);
  
  std::cout<<"number of poses after clustering "<<results.size()<<std::endl;
  //results = poseList;
  poses = results;
}

        

void PPFDetector::test()
{
    pcl::PointCloud<pcl::PointNormal>::Ptr model(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr sampled_model(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr transformed_model(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_model(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_scene(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    loadOBJ(model, "/home/yongqi/project/data/scene/Plate.obj", 50000, true);
    myppf::downSampling(model,sampled_model,7.5);
    
    colored_model->points.resize(sampled_model->points.size());
    for(int i=0; i<sampled_model->points.size(); i++)
    {
        colored_model->points[i].x = sampled_model->points[i].x;
        colored_model->points[i].y = sampled_model->points[i].y;
        colored_model->points[i].z = sampled_model->points[i].z;
        colored_model->points[i].r = 0;
        colored_model->points[i].g = 255;
        colored_model->points[i].b = 0;
    }
    
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::AngleAxisd rotation_vector(M_PI/3,Eigen::Vector3d(std::sqrt(3.0)/3.0,std::sqrt(3.0)/3.0,std::sqrt(3.0)/3.0));
    Eigen::Matrix3d rotation_matrix = rotation_vector.toRotationMatrix();
    T.block<3,3>(0,0) = rotation_matrix;
    T.topRightCorner<3,1>() = Eigen::Vector3d(50,25,50);
    cout<<T<<endl;
    transformed_model = sampled_model;
    transformPointCloud(transformed_model,T);
    sampled_scene = transformed_model;
    
    colored_scene->points.resize(transformed_model->points.size());
    for(int i=0; i<transformed_model->points.size(); i++)
    {
        colored_scene->points[i].x = transformed_model->points[i].x;
        colored_scene->points[i].y = transformed_model->points[i].y;
        colored_scene->points[i].z = transformed_model->points[i].z;
        colored_scene->points[i].r = 128;
        colored_scene->points[i].g = 128;
        colored_scene->points[i].b = 128;
    }
    
    
    /*std::cout<<sampled_scene->points.size()<<std::endl;
    std::vector<Pose3DPtr> list;
    match(list);
    std::cout<<"--------------------"<<std::endl;
    for(int i=0; i<list.size(); i++)
        std::cout<<list[i]->numVotes<<std::endl;
    save_poses("poses.txt");*/
    
 
    std::vector<Eigen::Matrix4d> pose_list;
    load_poses("poses.txt",pose_list);
    int n=0;
    for(int i=0; i<pose_list.size(); i++)
    {
        if(n>=0 && n<=0)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::transformPointCloud(*colored_model,*transformed,pose_list[i]);
            for(int i=0; i<transformed->points.size(); i++)
            {
                colored_scene->points.push_back(transformed->points[i]);
            }
        }
        n++;
    }
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("cloud viewer"));
    viewer->addPointCloud<pcl::PointXYZRGB>(colored_scene,"matching");
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> singleColor(sampled_pc, 0,255,0);
    //viewer->addPointCloud<pcl::PointNormal>(sampled_pc,singleColor,"cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "matching");
    //viewer->setBackgroundColor(0.0,0.0,0.0);
    //viewer->addPointCloudNormals<pcl::PointNormal,pcl::PointNormal>(SampledPC,SampledPC,1,2.0);
    //viewer->addPointCloudNormals<pcl::PointNormal,pcl::PointNormal>(sampled_scene,sampled_scene,1,1.0,"normal");
    
    while(!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

}
    
    
}
