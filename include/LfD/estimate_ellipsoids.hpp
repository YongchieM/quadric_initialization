#ifndef __LFD_ESTIMATE_ELLIPSOIDS_HPP__
#define __LFD_ESTIMATE_ELLIPSOIDS_HPP__

#include <iostream>
#include <math.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

namespace LfD
{

using eigMatrix = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;
using eigVector = Eigen::Matrix<double,Eigen::Dynamic,1>;


void dual_quadric_to_ellipsoid_parameters(Eigen::Matrix4d Q, Eigen::Vector3d& centre, Eigen::Vector3d& axes, Eigen::Matrix3d& R);
void sortEigenVectorByValues(eigVector& eigenValues, eigMatrix& eigenVectors);

void dual_ellipse_to_parameters(Eigen::Matrix3d C, Eigen::Vector2d& centre, Eigen::Vector2d& axes, Eigen::Matrix2d& R);


Eigen::Matrix3d fit_one_ellipse_in_bb(const Eigen::Vector4d bb);


void fit_ellipses_in_bbs(eigMatrix& bbs, eigMatrix& visibility, eigMatrix& Cs);

//build a 4x4 symmetric matrix from vector
Eigen::Matrix4d vector_to_symmetric_mat_4(const Eigen::Matrix<double,10,1> vec);

//serialize a symmetric 3x3 matrix
Eigen::Matrix<double,6,1> symmetric_mat_3_to_vector(const Eigen::Matrix3d C);

//Estimates one ellipsoid given projection matrices and detection ellipses for one image sequence
Eigen::Matrix4d estimate_one_ellipsoid(eigMatrix& Ps_t, eigMatrix& Cs);
Eigen::Matrix<double,6,10> compute_B(const Eigen::Matrix<double,4,3> P_fr);


void estimate_ellipsoids(eigMatrix& Ps_t, eigMatrix& input_ellipsoids_centres, eigMatrix& inputCs, eigMatrix& visibility, Eigen::Matrix<Eigen::Matrix4d,Eigen::Dynamic,1>& estQs);

//project the ellipsoids
void project_ellipsoids(eigMatrix& Ps_t, Eigen::Matrix<Eigen::Matrix4d,Eigen::Dynamic,1>& estQs, eigMatrix& visibility, eigMatrix& Cs);

//estimate one ellipsoid per object given detection bounding boxes and camera parameters
void compute_estimates(eigMatrix& bbs, const Eigen::Matrix3d K, eigMatrix& Ms_t, eigMatrix& visibility, eigMatrix& inputCs, eigMatrix& estCs, Eigen::Matrix<Eigen::Matrix4d,Eigen::Dynamic,1>& estQs_second_step);






}
#endif
