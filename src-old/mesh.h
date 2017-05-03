#include <Eigen/SVD>
#include <vcg/math/quadric.h>
#include <vcg/simplex/face/pos.h>
#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/smooth.h>
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>
#include <vcg/complex/algorithms/update/color.h>
#include <wrap/io_trimesh/import_ply.h>
#include <wrap/io_trimesh/export_ply.h>
#include <vcg/complex/algorithms/update/topology.h>
#include <vcg/complex/algorithms/update/normal.h>
#include <vcg/complex/algorithms/update/curvature.h>
#include <glog/logging.h>
#include "mlssurface.h"
#include "apss.h"
#include "implicits.h"
#include "apss.tpp"

class Vertex; class Edge; class Face;

struct UsedTypes: public vcg::UsedTypes<vcg::Use<::Vertex>::AsVertexType,
                                         vcg::Use<::Edge>::AsEdgeType,
                                         vcg::Use<::Face>::AsFaceType>{};

class Vertex: public vcg::Vertex<::UsedTypes,
                                 vcg::vertex::Coord3f,
                                 vcg::vertex::Normal3f,
                                 vcg::vertex::VFAdj,
                                 vcg::vertex::Mark,
                                 vcg::vertex::Radiusf,
                                 vcg::vertex::Curvaturef,
                                 vcg::vertex::CurvatureDirf,
                                 vcg::vertex::Qualityf,
                                 vcg::vertex::Color4b,
                                 vcg::vertex::BitFlags> {
    vcg::math::Quadric<double> _qd;
public:
    Vertex () {
        _qd.SetZero();
    }
	vcg::math::Quadric<double> &Qd () {
        return _qd;
    }
};

class Face: public vcg::Face<::UsedTypes,
                             vcg::face::VertexRef,
                             vcg::face::Mark,
                             vcg::face::VFAdj,
                             vcg::face::FFAdj,
                             vcg::face::BitFlags> {};
class Edge: public vcg::Edge<::UsedTypes> {};
class Mesh: public vcg::tri::TriMesh<std::vector<::Vertex>, std::vector<::Face>, std::vector<::Edge>> {};
typedef vcg::Point3<Mesh::ScalarType> Point3m;
typedef vcg::Matrix33<Mesh::ScalarType> Matrix33m;

class TriEdgeCollapse: public vcg::tri::TriEdgeCollapseQuadric<::Mesh, vcg::tri::BasicVertexPair<Vertex>, ::TriEdgeCollapse> {
public:
	class Params : public vcg::BaseParameterClass
	{
	public:
	  double    BoundaryQuadricWeight = 0.5;
	  bool      FastPreserveBoundary  = false;
	  bool      AreaCheck           = false;
	  bool      HardQualityCheck = false;
	  double    HardQualityThr = 0.1;
	  bool      HardNormalCheck =  false;
	  bool      NormalCheck           = false;
	  double    NormalThrRad          = M_PI/2.0;
	  double    CosineThr             = 0 ;
	  bool      OptimalPlacement =true;
	  bool      SVDPlacement = false;
	  bool      PreserveTopology =false;
	  bool      PreserveBoundary = false;
	  double    QuadricEpsilon = 1e-15;
	  bool      QualityCheck =true;
	  double    QualityThr =.3;     // Collapsed that generate faces with quality LOWER than this value are penalized. So higher the value -> better the quality of the accepted triangles
	  bool      QualityQuadric =false; // During the initialization manage all the edges as border edges adding a set of additional quadrics that are useful mostly for keeping face aspect ratio good.
	  double    QualityQuadricWeight = 0.001f; // During the initialization manage all the edges as border edges adding a set of additional quadrics that are useful mostly for keeping face aspect ratio good.
	  bool      QualityWeight=false;
	  double    QualityWeightFactor=100.0;
	  double    ScaleFactor=1.0;
	  bool      ScaleIndependent=true;
	  bool      UseArea =true;
	  bool      UseVertexWeight=false;
	};
    using TriEdgeCollapseQuadric::TriEdgeCollapseQuadric;
};

struct MeshModelParams {
    int smooth;
    float sample;
    MeshModelParams (): smooth(10), sample(10) {
    }
};

class MeshModel: public MeshModelParams {

    void colorize (Mesh &m) {
        GaelMls::APSS<Mesh> mls(m);
        LOG(INFO) << "Colorizing";
        mls.setFilterScale(3);
        mls.setMaxProjectionIters(15);
        mls.setProjectionAccuracy(0.0001);
        mls.setSphericalParameter(1);
        mls.computeVertexRaddi(16);

        // pass 1: computes curvatures and statistics
        for (auto &vert: m.vert) {

            Point3m p = mls.project(vert.P());
            float c = 0;

            int errorMask;
            Point3m grad = mls.gradient(p, &errorMask);
            if (errorMask == GaelMls::MLS_OK && grad.Norm() > 1e-8)
            {
                Matrix33m hess = mls.hessian(p, &errorMask);
                vcg::implicits::WeingartenMap<Mesh::ScalarType> W(grad,hess);
      
                vert.PD1() = W.K1Dir();
                vert.PD2() = W.K2Dir();
                vert.K1() =  W.K1();
                vert.K2() =  W.K2();

                c = W.MeanCurvature();
                /*
                switch(ct)
                {
                    case CT_MEAN: c = W.MeanCurvature(); break;
                    case CT_GAUSS: c = W.GaussCurvature(); break;
                    case CT_K1: c = W.K1(); break;
                    case CT_K2: c = W.K2(); break;
                    default: assert(0 && "invalid curvature type");
                }
                */
            }
            vert.Q() = c;
        }
        vcg::Histogramf H;
        vcg::tri::Stat<Mesh>::ComputePerVertexQualityHistogram(m,H);
        vcg::tri::UpdateColor<Mesh>::PerVertexQualityRamp(m,H.Percentile(0.01f),H.Percentile(0.99f));
    }
public:
    MeshModel (MeshModelParams const &params): MeshModelParams(params) {
    }

    void apply (Mesh &m) {
        /*
        vcg::tri::RequirePerVertexNormal(m);
        vcg::tri::RequirePerVertexMark(m);

        vcg::tri::UpdateNormal<Mesh>::PerVertexNormalized(m);
        vcg::tri::UpdateTopology<Mesh>::VertexFace(m);
        */
        LOG(INFO) << "Smoothing";
        vcg::tri::Smooth<Mesh>::VertexCoordLaplacian(m, smooth, false, true);   // smoothselected, cotangentWeight
        LOG(INFO) << "Simplifying";
        auto ovn = m.VN();
        auto ofn = m.FN();
        ::TriEdgeCollapse::Params params;
        vcg::LocalOptimization<Mesh> DeciSession(m, &params);
        DeciSession.Init<::TriEdgeCollapse>();
        int target = int(m.FN() / sample)+1;
        DeciSession.SetTargetSimplices(target);
        DeciSession.SetTimeBudget(0.5f);
        while( DeciSession.DoOptimization() && m.fn>target) {
          LOG(INFO)<< "Simplifying...";
        }
        DeciSession.Finalize<::TriEdgeCollapse >();
        int r = vcg::tri::Clean<Mesh>::RemoveDuplicateVertex(m);
        LOG(INFO) << "Removed " << r << " duplicated vertices";
        r = vcg::tri::Clean<Mesh>::RemoveUnreferencedVertex(m);
        LOG(INFO) << "Removed " << r << " unreferenced vertices";
        vcg::tri::Allocator<Mesh>::CompactEveryVector(m);
        LOG(INFO) << "Simplified, V: " << ovn << "->" << m.VN() << " F: " << ofn << "->" << m.FN();
        vcg::tri::Clean<Mesh>::FlipNormalOutside(m);
        /*
        vcg::tri::UpdateTopology<Mesh>::FaceFace(m);
        vcg::tri::UpdateCurvature<Mesh>::MeanAndGaussian(m);
        vcg::Histogramf H;
        vcg::tri::Stat<Mesh>::ComputePerVertexQualityHistogram(m,H);
        vcg::tri::UpdateColor<Mesh>::PerVertexQualityRamp(m,H.Percentile(0.01f),H.Percentile(0.99f));
        */
        vcg::tri::UpdateNormal<Mesh>::PerVertexNormalized(m);
        vcg::tri::UpdateTopology<Mesh>::VertexFace(m);
        colorize(m);
    }
};


namespace vcg { namespace tri { namespace io {
    int constexpr SAVE_MASK = Mask::IOM_VERTCOLOR | Mask::IOM_VERTQUALITY | Mask::IOM_VERTRADIUS |  vcg::tri::io::Mask::IOM_VERTNORMAL;
}}}

#include <wrap/ply/plylib.cpp>
#include "balltree.cpp"
