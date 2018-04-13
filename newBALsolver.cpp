#include <Eigen/Core>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "sophus/so3.h"


#include "tools/rotation.h"
#include "common/projection.h"

class VertexCameraBAL : public g2o::BaseVertex<9,Eigen::VectorXd>
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

    virtual void oplusImpl ( const double* update )
    {
        Eigen::Map<const Eigen::VectorXd> additive(update+3,6); //非const变量不能传参数给const变量
        Sophus::SO3 SO3_updated(update[0],update[1],update[2]);
        Sophus::SO3 SO3_estiamte(_estimate(0),_estimate(1),_estimate(2));
        Sophus::SO3 SO3_estiamted = SO3_updated * SO3_estiamte;
        Eigen::Vector3d so3_update = SO3_estiamted.log();
        _estimate.block(0,0,3,1) = so3_update;
        _estimate.block(3,0,6,6) = _estimate.block(3,0,6,6) + additive;
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

        ( *this ) ( cam->estimate().data(), point->estimate().data(), _error.data() );

    }

    template<typename T>
    bool operator() ( const T* camera, const T* point, T* residuals ) const
    {
        T predictions[2];
        CamProjectionWithDistortion ( camera, point, predictions );
        residuals[0] = predictions[0] - T ( measurement() ( 0 ) );
        residuals[1] = predictions[1] - T ( measurement() ( 1 ) );

        return true;
    }


    virtual void linearizeOplus() override
    {
        // use numeric Jacobians
        // BaseBinaryEdge<2, Vector2d, VertexCameraBAL, VertexPointBAL>::linearizeOplus();
        // return;

        // using autodiff from ceres. Otherwise, the system will use g2o numerical diff for Jacobians

        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );


        Eigen::Vector3d Pworld,Pcam;
        Sophus::SO3 rotation(cam->estimate()(0),cam->estimate()(1),cam->estimate()(2));
        Eigen::Map<const Eigen::Vector3d> translation(cam->estimate().data()+3,3);
        Pworld = point->estimate();
        Pcam = rotation * Pworld + translation;
        Eigen::Vector2d Pnorm;
        Pnorm(0) = -Pcam(0)/Pcam(2);
        Pnorm(1) = -Pcam(1)/Pcam(2);
        double r2 = Pnorm(0)*Pnorm(0) + Pnorm(1)*Pnorm(1);
        double focal,k1,k2;
        focal = cam->estimate()(6);
        k1 = cam->estimate()(7);
        k2 = cam->estimate()(8);
        //partial u to phi
        Eigen::Matrix<double,1,2> J_r2_Pnorm;
        Eigen::Matrix<double,2,3> J_Pnorm_Pcam , J_pixel_phi , J_pixel_Pcam;
        Eigen::Matrix3d J_Pcam_phi;
        J_r2_Pnorm(0,1) = 2*Pnorm(0);
        J_r2_Pnorm(0,2) = 2*Pnorm(1);
        J_Pnorm_Pcam(0,0) = -1.0/Pcam(2);
        J_Pnorm_Pcam(0,1) = 0;
        J_Pnorm_Pcam(0,2) = Pcam(0)/(Pcam(2)*Pcam(2));
        J_Pnorm_Pcam(1,0) = 0;
        J_Pnorm_Pcam(1,1) = -1.0/Pcam(2);
        J_Pnorm_Pcam(1,2) = Pcam(1)/(Pcam(2)*Pcam(2));
        J_Pcam_phi = -1.0*Sophus::SO3::hat(rotation*Pworld);
        J_pixel_Pcam = focal*(1+k1*r2+k2*r2*r2)*J_Pnorm_Pcam + focal*Pnorm*(k1+2*k2*r2)*J_r2_Pnorm*J_Pnorm_Pcam;
        J_pixel_phi = J_pixel_Pcam * J_Pcam_phi;
        //partial u to translation
        Eigen::Matrix<double,2,3> J_pixel_t;
        J_pixel_t = J_pixel_Pcam;
        //partial u to f,k1,k2
        Eigen::Vector2d J_pixel_f,J_pixel_k1,J_pixel_k2;
        J_pixel_f = (1+k1*r2+k2*r2*r2)*Pnorm;
        J_pixel_k1 = focal*r2*Pnorm;
        J_pixel_k2 = focal*r2*r2*Pnorm;
        //partial u to camprarmater
        _jacobianOplusXi.block(0,0,2,3) = J_pixel_phi;
        _jacobianOplusXi.block(0,3,2,3) = J_pixel_t;
        _jacobianOplusXi.block(0,6,2,1) = J_pixel_f;
        _jacobianOplusXi.block(0,7,2,1) = J_pixel_k1;
        _jacobianOplusXi.block(0,8,2,1) = J_pixel_k2;
        //partial u to point
        Eigen::Matrix3d J_Pcam_Pworld;
        J_Pcam_Pworld = rotation.matrix();
        _jacobianOplusXj = J_pixel_Pcam*J_Pcam_Pworld;

    }
};
