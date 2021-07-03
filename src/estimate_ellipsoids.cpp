#include "../include/LfD/estimate_ellipsoids.hpp"

namespace LfD
{
    
void dual_quadric_to_ellipsoid_parameters(Eigen::Matrix4d Q, Eigen::Vector3d& centre, Eigen::Vector3d& axes, Eigen::Matrix3d& R)
{
    Q /= (-Q(3,3));
    
    //compute ellipsoid centred on origin
    centre = -Q.topRightCorner<3,1>();
    Eigen::Matrix4d T;
    T << 1.0, 0.0, 0.0, -centre[0],
         0.0, 1.0, 0.0, -centre[1],
         0.0, 0.0, 1.0, -centre[2],
         0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix4d Qcent = T*Q*T.transpose();
    
    //compute axes and orientation
    Eigen::EigenSolver<eigMatrix> solver(Qcent.block<3,3>(0,0));
    eigVector D = solver.pseudoEigenvalueMatrix().diagonal();
    eigMatrix V = solver.pseudoEigenvectors();
    sortEigenVectorByValues(D,V);
    
    eigVector Dabs = D.array().abs();
    axes[0] = sqrt(Dabs[0]);
    axes[1] = sqrt(Dabs[1]);
    axes[2] = sqrt(Dabs[2]);
    
    R = V;
}

void sortEigenVectorByValues(eigVector& eigenValues, eigMatrix& eigenVectors) {
	std::vector<std::tuple<double,eigMatrix>> eigenValueAndVector;
	int size = static_cast<int>(eigenValues.size());
	
	eigenValueAndVector.reserve(size);
	for (int i = 0; i < size; ++i)
		eigenValueAndVector.push_back(std::tuple<double,eigMatrix>(eigenValues[i], eigenVectors.col(i)));
	
	//sort eigen values and vectors
	std::sort(eigenValueAndVector.begin(), eigenValueAndVector.end(),
		[&](const std::tuple<double,eigMatrix>& a, const std::tuple<double,eigMatrix>& b) -> bool {
		return std::get<0>(a) > std::get<0>(b);
		});
	
	for (int i = 0; i < size; ++i) {
		eigenValues[i] = std::get<0>(eigenValueAndVector[i]); //sorted eigen values
		eigenVectors.col(i).swap(std::get<1>(eigenValueAndVector[i])); //sorted eigen vectors
	}
}


void dual_ellipse_to_parameters(Eigen::Matrix3d C, Eigen::Vector2d& centre, Eigen::Vector2d& axes, Eigen::Matrix2d& R)
{
    if(C(2,2)>0)
        C /= (-C(2,2));
    
    centre = -C.topRightCorner<2,1>();
    
    Eigen::Matrix3d T;
    T << 1.0, 0.0, -centre[0],
         0.0, 1.0, -centre[1],
         0.0, 0.0, 1.0;
         
    Eigen::Matrix3d Ccent = T*C*T.transpose();
    Ccent = 0.5*(Ccent+Ccent.transpose());
    
    Eigen::EigenSolver<eigMatrix> solver(Ccent.block<2,2>(0,0));
    eigVector D = solver.pseudoEigenvalueMatrix().diagonal();
    eigMatrix V = solver.pseudoEigenvectors();
    
    axes[0] = sqrt(D.array().abs()[0]);
    axes[1] = sqrt(D.array().abs()[1]);
    R = V;
}

Eigen::Matrix3d fit_one_ellipse_in_bb(const Eigen::Vector4d bb)
{
    //encode ellipse size (axes)
    double width = abs(bb[2]-bb[0])/2.0;
    double height = abs(bb[3]-bb[1])/2.0;
    Eigen::Matrix3d Ccn;
    Ccn << 1.0/(width*width), 0.0, 0.0,
           0.0, 1.0/(height*height), 0.0,
           0.0, 0.0, -1.0;
    
    //encode ellipse location
    Eigen::Vector2d centre((bb[0]+bb[2])/2.0, (bb[1]+bb[3])/2.0);
    Eigen::Matrix3d P;
    P << 1.0, 0.0, centre[0],
         0.0, 1.0, centre[1],
         0.0, 0.0, 1.0;
    Eigen::Matrix3d Cinv = P*Ccn.inverse()*P.transpose();
    
    //force matrix to be symmetric
    Cinv = 0.5*(Cinv+Cinv.transpose());
    
    Eigen::Matrix3d C = Cinv/Cinv(2,2);
    
    if(C(0,0)+C(1,1) < 0)
        C = -C;
    
    return C;
}

void fit_ellipses_in_bbs(eigMatrix& bbs, eigMatrix& Cs)
{
    int n_frames = bbs.rows();
    
    Cs.resize(n_frames*3,3);
    Cs.setZero();
    
    for(int i=0; i<n_frames; i++)
    {
        Eigen::Vector4d bb = bbs.row(i).head(4);
        Eigen::Matrix3d C = fit_one_ellipse_in_bb(bb);
        Cs.block<3,3>(i*3,0) = C;
    }
               
}
    

Eigen::Matrix4d vector_to_symmetric_mat_4(const Eigen::Matrix<double,10,1> vec)
{
    Eigen::Matrix4d A;
    
    A(0,0) = vec[0];
    A(0,1) = vec[1];
    A(0,2) = vec[2];
    A(0,3) = vec[3];
    A(1,1) = vec[4];
    A(1,2) = vec[5];
    A(1,3) = vec[6];
    A(2,2) = vec[7];
    A(2,3) = vec[8];
    A(3,3) = vec[9];
    
    A(1,0) = vec[1];
    A(2,0) = vec[2];
    A(3,0) = vec[3];
    A(2,1) = vec[5];
    A(3,1) = vec[6];
    A(3,2) = vec[8];
    
    return A;
}


Eigen::Matrix<double,6,1> symmetric_mat_3_to_vector(const Eigen::Matrix3d C)
{
    Eigen::Matrix<double,6,1> vec;
    
    vec[0] = C(0,0);
    vec[1] = C(0,1);
    vec[2] = C(0,2);
    vec[3] = C(1,1);
    vec[4] = C(1,2);
    vec[5] = C(2,2);
    
    return vec;
}

Eigen::Matrix4d estimate_one_ellipsoid(eigMatrix& Ps_t, eigMatrix& Cs)
{
    int n_views = floor(Cs.rows()/3.0);
    
    eigMatrix M;
    M.resize(6*n_views, 10+n_views);
    M.setZero();
    
    Eigen::Vector2d centre,axes;
    Eigen::Matrix2d R;
    for(int i=0; i<n_views; i++)
    {
        //get centre and axes of current ellipse
        dual_ellipse_to_parameters(Cs.block<3,3>(3*i,0),centre,axes,R);
        
        //compute transformation used to precondition the ellipse: centre the ellipse and scale the axes
        double div_f = axes.norm();
        Eigen::Matrix3d T;
        T << div_f, 0.0, centre[0],
             0.0, div_f, centre[1],
             0.0, 0.0, 1.0;
        Eigen::Matrix3d T_t = T.inverse().transpose();
        
        //compute P_fr, applying T to the projection matrix
        Eigen::Matrix<double,4,3> P_fr = Ps_t.block<4,3>(4*i,0)*T_t;
        
        //compute the coefficients for the linear system
        Eigen::Matrix<double,6,10> B = compute_B(P_fr);
        
        //apply T to the ellipse
        Eigen::Matrix3d C_t = T.inverse()*Cs.block<3,3>(3*i,0)*T_t;
        
        //transform ellipse to vector form
        Eigen::Matrix<double,6,1> C_tv = symmetric_mat_3_to_vector(C_t);
        C_tv /= (-C_tv[5]);
        
        //write to M
        M.block<6,10>(6*i,0) = B;
        M.block<6,1>(6*i,10+i) = -C_tv;
    }
    
    Eigen::JacobiSVD<eigMatrix> svd(M,Eigen::ComputeThinU|Eigen::ComputeThinV);
    Eigen::Matrix<double,10,1> Qadjv = svd.matrixV().topRightCorner<10,1>();
    
    Eigen::Matrix4d adj_Q = vector_to_symmetric_mat_4(Qadjv);
    
    return adj_Q;
}

Eigen::Matrix<double,6,10> compute_B(const Eigen::Matrix<double,4,3> P_fr)
{
    Eigen::Matrix<double,6,10> B;
    B.setZero();
    
    Eigen::Matrix<double,9,1> r;
    r << P_fr(0,0),P_fr(0,1),P_fr(0,2),
         P_fr(1,0),P_fr(1,1),P_fr(1,2),
         P_fr(2,0),P_fr(2,1),P_fr(2,2);
    Eigen::Vector3d t(P_fr(3,0),P_fr(3,1),P_fr(3,2));
    
    B.row(0) << r[0]*r[0],
                2.0*r[3]*r[0],
                2.0*r[6]*r[0],
                2.0*t[0]*r[0],
                r[3]*r[3],
                2.0*r[6]*r[3],
                2.0*t[0]*r[3],
                r[6]*r[6],
                2.0*r[6]*t[0],
                t[0]*t[0];
            
    B.row(1) << r[1]*r[0],
                r[1]*r[3]+r[4]*r[0],
                r[7]*r[0]+r[1]*r[6],
                t[1]*r[0]+r[1]*t[0],
                r[4]*r[3],
                r[4]*r[6]+r[7]*r[3],
                r[4]*t[0]+t[1]*r[3],
                r[7]*r[6],
                r[7]*t[0]+t[1]*r[6],
                t[1]*t[0];
                
    B.row(2) << r[2]*r[0],
                r[2]*r[3]+r[5]*r[0],
                r[8]*r[0]+r[2]*r[6],
                t[2]*r[0]+r[2]*t[0],
                r[5]*r[3],
                r[5]*r[6]+r[8]*r[3],
                r[5]*t[0]+t[2]*r[3],
                r[8]*r[6],
                r[8]*t[0]+t[2]*r[6],
                t[2]*t[0];
                
    B.row(3) << r[1]*r[1],
                2.0*r[1]*r[4],
                2.0*r[1]*r[7],
                2.0*r[1]*t[1],
                r[4]*r[4],
                2.0*r[4]*r[7],
                2.0*r[4]*t[1],
                r[7]*r[7],
                2.0*r[7]*t[1],
                t[1]*t[1];
            
    B.row(4) << r[2]*r[1],
                r[2]*r[4]+r[5]*r[1],
                r[2]*r[7]+r[8]*r[1],
                t[2]*r[1]+r[2]*t[1],
                r[5]*r[4],
                r[5]*r[7]+r[8]*r[4],
                r[5]*t[1]+t[2]*r[4],
                r[8]*r[7],
                t[2]*r[7]+r[8]*t[1],
                t[2]*t[1];
                
    B.row(5) << r[2]*r[2],
                2.0*r[2]*r[5],
                2.0*r[2]*r[8],
                2.0*r[2]*t[2],
                r[5]*r[5],
                2.0*r[5]*r[8],
                2.0*r[5]*t[2],
                r[8]*r[8],
                2.0*r[8]*t[2],
                t[2]*t[2];
                
    return B;
}


void estimate_ellipsoid(eigMatrix& Ps_t, Eigen::Vector3d& input_ellipsoid_centre, eigMatrix& inputCs, Eigen::Matrix4d& estQ)
{
    
    if((inputCs.rows()/3) >= 3)
    {
        //compute the translation matrix due to the centre of the current ellipsoid
        Eigen::Matrix4d translM = Eigen::Matrix4d::Identity();
        translM.topRightCorner<3,1>() = input_ellipsoid_centre;
        
        //loop over the frames in witch the current object is present
        //apply the translation matrix to each projection matrix, for numerical preconditioning
        int instances = floor(Ps_t.rows()/4.0);
        for(int k=0; k<instances; k++)
        {
            Eigen::Matrix<double,3,4> first;
            first.setZero();
            first.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
            Eigen::Matrix4d second;
            second.block<3,4>(0,0) = Ps_t.block<4,3>(4*k,0).transpose();
            second.row(3) = Eigen::Vector4d(0.0,0.0,0.0,1.0);
            
            Ps_t.block<4,3>(4*k,0) = (first*second*translM).transpose();
        }
        
        //estimate the parameters of the current ellipsoid
        estQ = estimate_one_ellipsoid(Ps_t,inputCs);
        
        //re-apply the translation which had been removed
        estQ = translM*(estQ*translM.transpose());
        //force the estQ matrix to be symmmetric
        estQ = 0.5*(estQ+estQ.transpose());
        //scale the ellipsoid
        estQ /= (-estQ(3,3));
        
    }
    
    else
    {
        estQ.setZero();
    }

    
}


void project_ellipsoid(eigMatrix& Ps_t, Eigen::Matrix4d& estQ, eigMatrix& Cs)
{
    int n_frames = Ps_t.rows()/4.0;
    
    Cs.resize(3*n_frames,3);
    for(int i=0; i<n_frames; i++)
    {
       
        if(!estQ.isZero())
        {
            //transform the ellipsoid to the camera reference frame and project them
            Eigen::Matrix<double,3,4> P = Ps_t.block<4,3>(4*i,0).transpose();
            Eigen::Matrix3d Ctemp = P*estQ*P.transpose();
            
            //scale the ellipse
            Ctemp /= Ctemp(2,2);
            
            //store the results
            Cs.block<3,3>(3*i,0) = Ctemp;
        }
        else
        {
            Cs.block<3,3>(3*i,0) << 0, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0;
        }
    }
}

void compute_estimates(eigMatrix& bbs, const Eigen::Matrix3d K, eigMatrix& Ms_t, eigMatrix& inputCs, eigMatrix& estCs, Eigen::Matrix4d& estQ_second_step)
{
    //compute the stacked and transposed projection matrices
    eigMatrix Ps_t = (K*Ms_t.transpose()).transpose();
    
    //compute ellipses inscribed in the detection bounding boxes
    fit_ellipses_in_bbs(bbs,inputCs);
    
    //set the initial ellipsoids centre to the origin
    Eigen::Vector3d input_ellipsoid_centre;
    input_ellipsoid_centre.setZero();
    
    //perform the first round of estimation
    Eigen::Matrix4d estQ_first_step;
    estimate_ellipsoid(Ps_t,input_ellipsoid_centre,inputCs,estQ_first_step);
    
    //extract the centres of the current estimates for the ellipsoids
    Eigen::Vector3d first_step_ellipsoid_centre = input_ellipsoid_centre;
    first_step_ellipsoid_centre = estQ_first_step.topRightCorner<3,1>();

    //perform the second round of estimation
    estimate_ellipsoid(Ps_t,first_step_ellipsoid_centre,inputCs,estQ_second_step);
    
    //project the estimated ellipsoids
    project_ellipsoid(Ps_t,estQ_second_step,estCs);
}


void convertPose(Eigen::MatrixXd& in, Eigen::MatrixXd& out)
{
    int rows = in.rows();
    int cols = in.cols();
    out.resize(4*rows,3);

    for(int i=0; i<rows; i++)
    {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.topRightCorner<3,1>() = in.row(i).head(3);
        Eigen::Quaterniond q(in.row(i)[6],in.row(i)[3],in.row(i)[4],in.row(i)[5]);
        T.block<3,3>(0,0) = q.toRotationMatrix();
        
        out.block<4,3>(4*i,0) = T.inverse().block<3,4>(0,0).transpose();
    }
}


/**************************************************/
Eigen::Matrix4d generate_ellipsoid_from_ellipses(eigMatrix& pose, eigMatrix& bbs, const Eigen::Matrix3d K)
{
    Eigen::MatrixXd inputCs,estCs;
    Eigen::MatrixXd Ms_t;
    
    convertPose(pose,Ms_t);
    
    Eigen::Matrix4d estQ;
    
    compute_estimates(bbs,K,Ms_t,inputCs,estCs,estQ);
    
    return estQ;
}



}

