#include "fusion/fusion.h"
#include "common/io.h"

#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <tuple>

namespace dpe {
namespace {

struct Vec3 { float x,y,z; };
Vec3 CamPoint(const Camera& c,float x,float y,float d){return {d*(x-c.K[2])/c.K[0],d*(y-c.K[5])/c.K[4],d};}
Vec3 CamToWorld(const Camera& c,const Vec3& X){
    const float a=X.x-c.t[0],b=X.y-c.t[1],d=X.z-c.t[2];
    return {c.R[0]*a+c.R[3]*b+c.R[6]*d,c.R[1]*a+c.R[4]*b+c.R[7]*d,c.R[2]*a+c.R[5]*b+c.R[8]*d};
}
Vec3 WorldToCam(const Camera& c,const Vec3& X){return {c.R[0]*X.x+c.R[1]*X.y+c.R[2]*X.z+c.t[0],c.R[3]*X.x+c.R[4]*X.y+c.R[5]*X.z+c.t[1],c.R[6]*X.x+c.R[7]*X.y+c.R[8]*X.z+c.t[2]};}
cv::Point2f Project(const Camera& c,const Vec3& X){
    const float z=c.K[6]*X.x+c.K[7]*X.y+c.K[8]*X.z;
    return {(c.K[0]*X.x+c.K[1]*X.y+c.K[2]*X.z)/z,
            (c.K[3]*X.x+c.K[4]*X.y+c.K[5]*X.z)/z};
}
float NormalAngle(const cv::Vec3f& a,const cv::Vec3f& b){
    const float na=std::sqrt(a.dot(a)),nb=std::sqrt(b.dot(b));if(na<1e-8f||nb<1e-8f)return 3.14159f;
    float d=a.dot(b)/(na*nb);d=std::max(-1.0f,std::min(1.0f,d));return std::acos(d);
}

struct PlyPoint { Vec3 p; cv::Vec3b c; };

}  // namespace

void RunFusion(Scene& scene,
               const ReconstructionState& reconstruction,
               const std::vector<Problem>& problems,
               const boost::filesystem::path& output_ply) {
    std::vector<PlyPoint> cloud;
    std::unordered_map<int, cv::Mat> used;
    for (const auto& p : problems) {
        const FrameState* ref_state=reconstruction.Find(p.ref_image_id);
        if(!ref_state||ref_state->depth.empty())continue;
        const cv::Mat& ref_img=scene.ColorImage(p.ref_image_id);
        Camera ref_cam=scene.ScaledCamera(p.ref_image_id,ref_state->depth.cols,ref_state->depth.rows);
        if(used.find(p.ref_image_id)==used.end())used[p.ref_image_id]=cv::Mat(ref_state->depth.size(),CV_8U,cv::Scalar(0));

        for(int y=0;y<ref_state->depth.rows;++y){
            for(int x=0;x<ref_state->depth.cols;++x){
                if(used[p.ref_image_id].at<unsigned char>(y,x))continue;
                const float d=ref_state->depth.at<float>(y,x);if(d<=0)continue;
                const Vec3 Xw=CamToWorld(ref_cam,CamPoint(ref_cam,x,y,d));
                const cv::Vec3f nr=ref_state->normal.at<cv::Vec3f>(y,x);
                int consistent=0;float dynamic=0.0f;
                const int rx=std::min(ref_img.cols-1,std::max(0,static_cast<int>(x*ref_img.cols/static_cast<float>(ref_state->depth.cols))));
                const int ry=std::min(ref_img.rows-1,std::max(0,static_cast<int>(y*ref_img.rows/static_cast<float>(ref_state->depth.rows))));
                const cv::Vec3b rc=ref_img.at<cv::Vec3b>(ry,rx);
                cv::Vec3i color_sum(rc[0],rc[1],rc[2]);
                std::vector<std::tuple<int,int,int>> matches;

                for(int sid:p.src_image_ids){
                    const FrameState* src_state=reconstruction.Find(sid);if(!src_state||src_state->depth.empty())continue;
                    Camera sc=scene.ScaledCamera(sid,src_state->depth.cols,src_state->depth.rows);
                    const Vec3 Xs=WorldToCam(sc,Xw);if(Xs.z<=0)continue;
                    const cv::Point2f q=Project(sc,Xs);const int qx=static_cast<int>(std::round(q.x)),qy=static_cast<int>(std::round(q.y));
                    if(qx<0||qy<0||qx>=src_state->depth.cols||qy>=src_state->depth.rows)continue;
                    const float sd=src_state->depth.at<float>(qy,qx);if(sd<=0)continue;
                    const Vec3 Xw2=CamToWorld(sc,CamPoint(sc,qx,qy,sd));
                    const Vec3 Xr2=WorldToCam(ref_cam,Xw2);if(Xr2.z<=0)continue;
                    const cv::Point2f rp=Project(ref_cam,Xr2);
                    const float reproj=std::hypot(rp.x-x,rp.y-y);
                    const float rel_depth=std::fabs(Xr2.z-d)/std::max(d,1e-6f);
                    const float angle=NormalAngle(nr,src_state->normal.at<cv::Vec3f>(qy,qx));
                    if(reproj<2.0f&&rel_depth<0.01f&&angle<0.174533f){
                        ++consistent;
                        dynamic+=std::exp(-(reproj+200.0f*rel_depth+10.0f*angle));
                        matches.emplace_back(sid,qx,qy);
                        const cv::Mat& si=scene.ColorImage(sid);
                        const int ix=std::min(si.cols-1,std::max(0,static_cast<int>(qx*si.cols/static_cast<float>(src_state->depth.cols))));
                        const int iy=std::min(si.rows-1,std::max(0,static_cast<int>(qy*si.rows/static_cast<float>(src_state->depth.rows))));
                        const cv::Vec3b cc=si.at<cv::Vec3b>(iy,ix);
                        color_sum[0]+=cc[0]; color_sum[1]+=cc[1]; color_sum[2]+=cc[2];
                    }
                }
                const float factor=ref_state->reliability.at<unsigned char>(y,x)==WEAK?0.45f:0.30f;
                if(consistent>=1&&dynamic>factor*consistent){
                    cv::Vec3b color(static_cast<unsigned char>(color_sum[0]/(consistent+1)),
                                    static_cast<unsigned char>(color_sum[1]/(consistent+1)),
                                    static_cast<unsigned char>(color_sum[2]/(consistent+1)));
                    cloud.push_back({Xw,color});
                    used[p.ref_image_id].at<unsigned char>(y,x)=1;
                    for(const auto& m:matches){
                        const int sid=std::get<0>(m),qx=std::get<1>(m),qy=std::get<2>(m);
                        const FrameState* ss=reconstruction.Find(sid);
                        if(used.find(sid)==used.end())used[sid]=cv::Mat(ss->depth.size(),CV_8U,cv::Scalar(0));
                        used[sid].at<unsigned char>(qy,qx)=1;
                    }
                }
            }
        }
    }

    std::ofstream out(output_ply.string());
    out<<"ply\nformat ascii 1.0\nelement vertex "<<cloud.size()<<"\n"
       <<"property float x\nproperty float y\nproperty float z\n"
       <<"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for(const auto& v:cloud)out<<v.p.x<<' '<<v.p.y<<' '<<v.p.z<<' '<<(int)v.c[2]<<' '<<(int)v.c[1]<<' '<<(int)v.c[0]<<'\n';
    std::cout<<"fusion: "<<cloud.size()<<" points -> "<<output_ply.string()<<"\n";
}

}  // namespace dpe
