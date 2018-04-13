#include <Eigen/Core>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "sophus/so3.h"
#include "sophus/se3.h"
#include "ceres/autodiff.h"

#include "tools/rotation.h"
#include "common/projection.h"
class CameraSO3{
public:
    CameraSO3(){}
    CameraSO3(Sophus::SO3 so3,Eigen::Vector3d t,double f,double k1,double k2):so3_(so3),t_(t),f_(f),k1_(k1),k2_(k2){

    }
public:
    Sophus::SO3 so3_;
    Eigen::Vector3d t_;
    double f_;
    double k1_;
    double k2_;
};

class VertexCameraBAL : public g2o::BaseVertex<9,CameraSO3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update ) {
        /*Eigen::VectorXd::ConstMapType v(update, VertexCameraBAL::Dimension);
        _estimate += v;*/

        //so3
      /*   Eigen::Map<const Eigen::VectorXd> additive(update+3,6); //非const变量不能传参数给const变量
         Sophus::SO3 SO3_updated(update[0],update[1],update[2]);
         Sophus::SO3 SO3_estiamte(_estimate(0),_estimate(1),_estimate(2));
         Sophus::SO3 SO3_estiamted = SO3_updated * SO3_estiamte;
         Eigen::Vector3d so3_update = SO3_estiamted.log();
         Eigen::Matrix<double,9,1> update_estimate;
         update_estimate.block(0,0,3,1) = so3_update;
         update_estimate.block(3,0,6,1) = _estimate.block(3,0,6,1) + additive;
         setEstimate(update_estimate);*/

        Sophus::SO3 SO3_up(update[0],update[1],update[2]);
        Eigen::Vector3d t_up(update[3],update[4],update[5]);
        _estimate.so3_ = SO3_up*_estimate.so3_;
        _estimate.t_ += t_up;
        _estimate.f_ += update[6];
        _estimate.k1_ += update[7];
        _estimate.k2_ += update[8];




    }

};


class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        Eigen::Vector3d::ConstMapType v ( update );
        _estimate += v;
    }
};

class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d,VertexCameraBAL, VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void computeError() override   // The virtual function comes from the Edge base class. Must define if you use edge.
    {
        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );

        //( *this ) ( cam->estimate().data(), point->estimate().data(), _error.data() );

        Eigen::Vector3d P_world = point->estimate();
        Eigen::Vector3d P_cam = cam->estimate().so3_*P_world + cam->estimate().t_;
        Eigen::Vector2d P_norm;
        P_norm(0) = -P_cam(0)/P_cam(2);
        P_norm(1) = -P_cam(1)/P_cam(2);
        const double& f = cam->estimate().f_;
        const double& k1 = cam->estimate().k1_;
        const double& k2 = cam->estimate().k2_;
        double r2 = P_norm(0)*P_norm(0) + P_norm(1)*P_norm(1);
        Eigen::Vector2d predictions;
        predictions = f*(1 + k1*r2 + k2*r2*r2)*P_norm;
        //Eigen::Vector2d tt(_error);
        _error = predictions - measurement();
        Eigen::Vector2d tt(_error);

    }

    /*template<typename T>
    bool operator() ( const T* camera, const T* point, T* residuals ) const
    {
        T predictions[2];
        CamProjectionWithDistortion ( camera, point, predictions );
        residuals[0] = predictions[0] - T ( measurement() ( 0 ) );
        residuals[1] = predictions[1] - T ( measurement() ( 1 ) );

        return true;
    }*/


    virtual void linearizeOplus() override {
        // use numeric Jacobians
        /* BaseBinaryEdge<2, Eigen::Vector2d, VertexCameraBAL, VertexPointBAL>::linearizeOplus();
         return;*/
        // using autodiff from ceres. Otherwise, the system will use g2o numerical diff for Jacobians

        const VertexCameraBAL *cam = static_cast<const VertexCameraBAL *> ( vertex(0));
        const VertexPointBAL *point = static_cast<const VertexPointBAL *> ( vertex(1));

        Eigen::Vector3d Pworld, Pcam;
        Sophus::SO3 rotation(cam->estimate().so3_);
        Eigen::Vector3d translation(cam->estimate().t_);
        Pworld = point->estimate();
        Pcam = rotation * Pworld + translation;
        Eigen::Vector2d Pnorm;
        Pnorm(0) = -Pcam(0) / Pcam(2);
        Pnorm(1) = -Pcam(1) / Pcam(2);
        double r2 = Pnorm(0) * Pnorm(0) + Pnorm(1) * Pnorm(1);
        double focal, k1, k2;
        focal = cam->estimate().f_;
        k1 = cam->estimate().k1_;
        k2 = cam->estimate().k2_;
        //partial u to phi
        Eigen::Matrix<double, 1, 2> J_r2_Pnorm;
        Eigen::Matrix<double, 2, 3> J_Pnorm_Pcam, J_pixel_phi, J_pixel_Pcam;
        Eigen::Matrix3d J_Pcam_phi;
        J_r2_Pnorm(0, 0) = 2 * Pnorm(0);
        J_r2_Pnorm(0, 1) = 2 * Pnorm(1);
        J_Pnorm_Pcam(0, 0) = -1.0 / Pcam(2);
        J_Pnorm_Pcam(0, 1) = 0;
        J_Pnorm_Pcam(0, 2) = Pcam(0) / (Pcam(2) * Pcam(2));
        J_Pnorm_Pcam(1, 0) = 0;
        J_Pnorm_Pcam(1, 1) = -1.0 / Pcam(2);
        J_Pnorm_Pcam(1, 2) = Pcam(1) / (Pcam(2) * Pcam(2));
        J_Pcam_phi = -1.0 * Sophus::SO3::hat(rotation * Pworld);
        J_pixel_Pcam = focal * (1 + k1 * r2 + k2 * r2 * r2) * J_Pnorm_Pcam +
                       focal * Pnorm * (k1 + 2 * k2 * r2) * J_r2_Pnorm * J_Pnorm_Pcam;
        J_pixel_phi = J_pixel_Pcam * J_Pcam_phi;
        //partial u to translation
        Eigen::Matrix<double, 2, 3> J_pixel_t;
        J_pixel_t = J_pixel_Pcam;
        //partial u to f,k1,k2
        Eigen::Vector2d J_pixel_f, J_pixel_k1, J_pixel_k2;
        J_pixel_f = (1 + k1 * r2 + k2 * r2 * r2) * Pnorm;
        J_pixel_k1 = focal * r2 * Pnorm;
        J_pixel_k2 = focal * r2 * r2 * Pnorm;
        //partial u to camprarmater
        _jacobianOplusXi.block(0, 0, 2, 3) = J_pixel_phi;
        _jacobianOplusXi.block(0, 3, 2, 3) = J_pixel_t;
        _jacobianOplusXi.block(0, 6, 2, 1) = J_pixel_f;
        _jacobianOplusXi.block(0, 7, 2, 1) = J_pixel_k1;
        _jacobianOplusXi.block(0, 8, 2, 1) = J_pixel_k2;
        //partial u to point
        Eigen::Matrix3d J_Pcam_Pworld;
        J_Pcam_Pworld = rotation.matrix();
        _jacobianOplusXj = J_pixel_Pcam * J_Pcam_Pworld;


         /* typedef ceres::internal::AutoDiff<EdgeObservationBAL, double, VertexCameraBAL::Dimension, VertexPointBAL::Dimension> BalAutoDiff;

          Eigen::Matrix<double, Dimension, VertexCameraBAL::Dimension, Eigen::RowMajor> dError_dCamera;
          Eigen::Matrix<double, Dimension, VertexPointBAL::Dimension, Eigen::RowMajor> dError_dPoint;
          double *parameters[] = { const_cast<double*> ( cam->estimate().data() ), const_cast<double*> ( point->estimate().data() ) };
          double *jacobians[] = { dError_dCamera.data(), dError_dPoint.data() };
          double value[Dimension];
          bool diffState = BalAutoDiff::Differentiate ( *this, parameters, Dimension, value, jacobians );

          // copy over the Jacobians (convert row-major -> column-major)
          if ( diffState )
          {
              _jacobianOplusXi = dError_dCamera;
              _jacobianOplusXj = dError_dPoint;
          }
          else
          {
              assert ( 0 && "Error while differentiating" );
              _jacobianOplusXi.setZero();
              _jacobianOplusXi.setZero();
          }*/
    }
};
