#include "studio/artifact_writer.h"
#include "common/io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace dpe {
namespace {
std::string Escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '\\' || c == '"') out.push_back('\\');
        if (c == '\n') out += "\\n"; else out.push_back(c);
    }
    return out;
}

void WriteFloatArray(std::ostream& os, const float* v, int n) {
    os << '[';
    for (int i = 0; i < n; ++i) { if (i) os << ','; os << std::setprecision(9) << v[i]; }
    os << ']';
}

cv::Mat ScalarPreview(const cv::Mat& src, const cv::Mat& valid = cv::Mat(), float fixed_max = -1.0f) {
    if (src.empty()) return {};
    cv::Mat f;
    src.convertTo(f, CV_32F);
    double lo = 0.0, hi = fixed_max;
    if (fixed_max <= 0) {
        std::vector<float> values;
        values.reserve(std::min<size_t>(src.total(), 200000));
        const size_t step = std::max<size_t>(1, src.total() / 200000);
        const float* p = f.ptr<float>();
        for (size_t i=0;i<src.total();i+=step) {
            if (!valid.empty() && valid.ptr<unsigned char>()[i] == 0) continue;
            if (std::isfinite(p[i]) && p[i] >= 0) values.push_back(p[i]);
        }
        if (!values.empty()) {
            const size_t k = static_cast<size_t>(0.98 * (values.size()-1));
            std::nth_element(values.begin(), values.begin()+k, values.end());
            hi = std::max(1e-6, static_cast<double>(values[k]));
        } else hi = 1.0;
    }
    cv::Mat u8(f.size(), CV_8U, cv::Scalar(0));
    for (int y=0;y<f.rows;++y) for(int x=0;x<f.cols;++x) {
        if (!valid.empty() && !valid.at<unsigned char>(y,x)) continue;
        const float v=f.at<float>(y,x);
        if(!std::isfinite(v))continue;
        const float t=std::max(0.0f,std::min(1.0f,static_cast<float>((v-lo)/(hi-lo+1e-12))));
        u8.at<unsigned char>(y,x)=static_cast<unsigned char>(std::lround(t*255));
    }
    cv::Mat color;
    cv::applyColorMap(u8,color,cv::COLORMAP_TURBO);
    if(!valid.empty()) color.setTo(cv::Scalar(20,20,20), valid==0);
    return color;
}

cv::Mat RatioPreview(const cv::Mat& num, const cv::Mat& den) {
    cv::Mat ratio(num.size(), CV_32F, cv::Scalar(0));
    for(int y=0;y<num.rows;++y)for(int x=0;x<num.cols;++x){
        const float d=den.type()==CV_8U?den.at<unsigned char>(y,x):static_cast<float>(den.at<int>(y,x));
        const float n=num.type()==CV_8U?num.at<unsigned char>(y,x):static_cast<float>(num.at<int>(y,x));
        if(d>0)ratio.at<float>(y,x)=n/d;
    }
    return ScalarPreview(ratio,cv::Mat(),1.0f);
}

cv::Mat SurfacePreview(const cv::Mat& labels) {
    cv::Mat out(labels.size(),CV_8UC3,cv::Scalar(0,0,0));
    for(int y=0;y<labels.rows;++y)for(int x=0;x<labels.cols;++x){
        const uint32_t v=static_cast<uint32_t>(std::max(0,labels.at<int>(y,x)));
        if(!v)continue;
        uint32_t h=v*2654435761u;
        out.at<cv::Vec3b>(y,x)=cv::Vec3b(60+(h&127),60+((h>>8)&127),60+((h>>16)&127));
    }
    return out;
}

cv::Mat EdgeRelationPreview(const cv::Mat& image_edge,const cv::Mat& gt_edge,const cv::Mat& valid){
    cv::Mat out(image_edge.size(),CV_8UC3,cv::Scalar(18,18,18));
    for(int y=0;y<out.rows;++y)for(int x=0;x<out.cols;++x){
        if(!valid.empty()&&!valid.at<unsigned char>(y,x))continue;
        const bool a=image_edge.at<unsigned char>(y,x)!=0,b=gt_edge.at<unsigned char>(y,x)!=0;
        if(a&&b)out.at<cv::Vec3b>(y,x)=cv::Vec3b(80,210,90);      // green: image+geometry
        else if(a)out.at<cv::Vec3b>(y,x)=cv::Vec3b(70,70,235);     // red in RGB: image-only
        else if(b)out.at<cv::Vec3b>(y,x)=cv::Vec3b(230,130,60);    // blue/orange: missed geometry edge
    }
    return out;
}

void WriteSparse(const boost::filesystem::path& dir, const FrameTelemetry& t) {
    if(!t.anchors.empty()){
        std::ofstream out((dir/"anchors.bin").string(),std::ios::binary);
        const uint32_t magic=0x44504131u, count=static_cast<uint32_t>(t.anchors.size());
        out.write(reinterpret_cast<const char*>(&magic),4);out.write(reinterpret_cast<const char*>(&count),4);
        for(const auto& r:t.anchors){out.write(reinterpret_cast<const char*>(&r.pixel_index),4);out.write(reinterpret_cast<const char*>(r.xy),sizeof(r.xy));}
    }
    if(!t.planes.empty()){
        std::ofstream out((dir/"planes.bin").string(),std::ios::binary);
        const uint32_t magic=0x44505031u, count=static_cast<uint32_t>(t.planes.size());
        out.write(reinterpret_cast<const char*>(&magic),4);out.write(reinterpret_cast<const char*>(&count),4);
        for(const auto& r:t.planes){out.write(reinterpret_cast<const char*>(&r.pixel_index),4);out.write(reinterpret_cast<const char*>(r.plane),sizeof(r.plane));out.write(reinterpret_cast<const char*>(&r.radius),4);}
    }
}

struct SummaryRow {
    int id=0; long valid=0; double mean_err=0, under2=0, under10=0;
    double candidate_same=0, anchor_same=0, plane_depth=0, plane_normal=0, radius_violation=0;
};

SummaryRow Summarize(int id,const FrameTelemetry& t){
    SummaryRow r;r.id=id;
    if(t.Empty())return r;
    double errsum=0, cand=0,candsame=0,anchors=0,anchorsame=0,pde=0,pne=0,rv=0;long pc=0,pn=0,rc=0;
    for(int y=0;y<t.gt_valid.rows;++y)for(int x=0;x<t.gt_valid.cols;++x){
        if(!t.gt_valid.at<unsigned char>(y,x))continue;
        ++r.valid;
        const float e=t.final_depth_error.at<float>(y,x);if(e>=0&&std::isfinite(e)){errsum+=e;if(e<=0.02)r.under2++;if(e<=0.10)r.under10++;}
        const int c=t.candidate_count.at<int>(y,x), cs=t.same_surface_candidates.at<int>(y,x);cand+=c;candsame+=cs;
        const int a=t.anchor_count.at<unsigned char>(y,x),as=t.same_surface_anchors.at<unsigned char>(y,x);anchors+=a;anchorsame+=as;
        const float de=t.fitted_plane_depth_error.at<float>(y,x);if(de>0&&std::isfinite(de)){pde+=de;++pc;}
        const float ne=t.fitted_plane_normal_error.at<float>(y,x);if(ne>0&&std::isfinite(ne)){pne+=ne;++pn;}
        const float v=t.radius_violation.at<float>(y,x);if(v>=0&&std::isfinite(v)){rv+=v;++rc;}
    }
    if(r.valid){r.mean_err=errsum/r.valid;r.under2/=r.valid;r.under10/=r.valid;}
    r.candidate_same=cand>0?candsame/cand:0;r.anchor_same=anchors>0?anchorsame/anchors:0;
    r.plane_depth=pc?pde/pc:0;r.plane_normal=pn?pne/pn:0;r.radius_violation=rc?rv/rc:0;
    return r;
}
}

ArtifactWriter::ArtifactWriter(const boost::filesystem::path& experiment_root,
                               const boost::filesystem::path& dense_root)
    : root_(experiment_root), dense_(dense_root) { boost::filesystem::create_directories(root_); }

void ArtifactWriter::WriteStatus(const std::string& case_name, const std::string& phase,
                                 int completed, int total, const std::string& message) {
    std::ofstream out((root_ / "status.json").string());
    out << "{\n  \"case\":\"" << Escape(case_name) << "\",\n  \"phase\":\"" << Escape(phase)
        << "\",\n  \"completed\":" << completed << ",\n  \"total\":" << total
        << ",\n  \"message\":\"" << Escape(message) << "\"\n}\n";
}

void ArtifactWriter::WriteFinalViews(const std::string& case_name,
                                     const std::vector<Problem>& problems,
                                     const ReconstructionState& reconstruction) {
    const auto base = root_ / case_name / "views";
    boost::filesystem::create_directories(base);
    for (const auto& p : problems) {
        const FrameState* s = reconstruction.Find(p.ref_image_id);
        if (!s || s->Empty()) continue;
        const auto dir = base / FormatIndex(p.ref_image_id);
        boost::filesystem::create_directories(dir);
        WriteBinMat(dir / "depth.dmb", s->depth); WriteBinMat(dir / "normal.dmb", s->normal);
        WriteBinMat(dir / "state.dmb", s->reliability); WriteBinMat(dir / "selected_views.dmb", s->selected_views);
        WriteDepthPreview(dir / "depth.jpg", s->depth, p.params.depth_min, p.params.depth_max);
        WriteNormalPreview(dir / "normal.jpg", s->normal); WriteReliabilityPreview(dir / "state.jpg", s->reliability);

        const auto& t=s->telemetry;
        if(t.Empty())continue;
        WriteBinMat(dir/"gt_depth.dmb",t.gt_depth);WriteBinMat(dir/"gt_normal.dmb",t.gt_normal);
        WriteBinMat(dir/"gt_valid.dmb",t.gt_valid);WriteBinMat(dir/"gt_geometry_edge.dmb",t.gt_geometry_edge);
        WriteBinMat(dir/"gt_surface_label.dmb",t.gt_surface_label);
        WriteBinMat(dir/"fine_edge.dmb",t.fine_edge);WriteBinMat(dir/"coarse_region.dmb",t.coarse_region);WriteBinMat(dir/"texture_complexity.dmb",t.texture_complexity);
        WriteBinMat(dir/"es_candidate_count.dmb",t.es_candidate_count);WriteBinMat(dir/"es_same_surface.dmb",t.es_same_surface);
        WriteBinMat(dir/"candidate_count.dmb",t.candidate_count);
        WriteBinMat(dir/"same_surface_candidates.dmb",t.same_surface_candidates);WriteBinMat(dir/"anchor_count.dmb",t.anchor_count);
        WriteBinMat(dir/"same_surface_anchors.dmb",t.same_surface_anchors);WriteBinMat(dir/"plane_depth_error.dmb",t.fitted_plane_depth_error);
        WriteBinMat(dir/"plane_normal_error.dmb",t.fitted_plane_normal_error);WriteBinMat(dir/"radius_violation.dmb",t.radius_violation);
        WriteBinMat(dir/"final_depth_error.dmb",t.final_depth_error);WriteBinMat(dir/"matching_cost.dmb",t.matching_cost);
        WriteBinMat(dir/"adaptive_radius.dmb",t.adaptive_radius);
        WriteDepthPreview(dir/"gt_depth.jpg",t.gt_depth,p.params.depth_min,p.params.depth_max);
        WriteNormalPreview(dir/"gt_normal.jpg",t.gt_normal);
        cv::imwrite((dir/"gt_geometry_edge.png").string(),t.gt_geometry_edge);
        cv::imwrite((dir/"edge_relation.png").string(),EdgeRelationPreview(t.fine_edge,t.gt_geometry_edge,t.gt_valid));
        cv::imwrite((dir/"gt_surface_label.png").string(),SurfacePreview(t.gt_surface_label));
        cv::imwrite((dir/"fine_edge.png").string(),t.fine_edge);
        cv::imwrite((dir/"coarse_region.png").string(),SurfacePreview(t.coarse_region));
        cv::imwrite((dir/"texture_complexity.png").string(),ScalarPreview(t.texture_complexity,cv::Mat(),1.0f));
        cv::imwrite((dir/"es_candidate_count.png").string(),ScalarPreview(t.es_candidate_count));
        cv::imwrite((dir/"es_same_ratio.png").string(),RatioPreview(t.es_same_surface,t.es_candidate_count));
        cv::imwrite((dir/"candidate_count.png").string(),ScalarPreview(t.candidate_count));
        cv::imwrite((dir/"candidate_same_ratio.png").string(),RatioPreview(t.same_surface_candidates,t.candidate_count));
        cv::imwrite((dir/"anchor_same_ratio.png").string(),RatioPreview(t.same_surface_anchors,t.anchor_count));
        cv::imwrite((dir/"plane_depth_error.png").string(),ScalarPreview(t.fitted_plane_depth_error,t.gt_valid));
        cv::imwrite((dir/"plane_normal_error.png").string(),ScalarPreview(t.fitted_plane_normal_error,t.gt_valid,45.0f));
        cv::imwrite((dir/"radius_violation.png").string(),ScalarPreview(t.radius_violation,t.gt_valid,1.0f));
        cv::imwrite((dir/"final_depth_error.png").string(),ScalarPreview(t.final_depth_error,t.gt_valid,0.10f));
        cv::imwrite((dir/"matching_cost.png").string(),ScalarPreview(t.matching_cost,cv::Mat(),2.0f));
        cv::imwrite((dir/"adaptive_radius.png").string(),ScalarPreview(t.adaptive_radius));
        WriteSparse(dir,t);
    }
}

boost::filesystem::path ArtifactWriter::WriteGroundTruthPreview(const GroundTruthProvider& ground_truth) {
    const auto path = root_ / "_ground_truth" / "aligned_scan_preview.ply";
    if (!boost::filesystem::exists(path)) ground_truth.WritePreviewPly(path);
    return path;
}

void ArtifactWriter::WriteTelemetrySummary(const std::string& case_name,
                                           const std::vector<Problem>& problems,
                                           const ReconstructionState& reconstruction){
    const auto dir=root_/case_name;boost::filesystem::create_directories(dir);
    std::ofstream csv((dir/"telemetry_summary.csv").string());
    csv<<"view_id,gt_valid_pixels,mean_depth_error_m,depth_within_2cm,depth_within_10cm,candidate_same_surface_ratio,anchor_same_surface_ratio,mean_plane_depth_error_m,mean_plane_normal_error_deg,mean_radius_violation\n";
    std::ofstream json((dir/"telemetry_summary.json").string());json<<"{\n  \"views\":[\n";
    bool first=true;
    for(const auto& p:problems){const FrameState* s=reconstruction.Find(p.ref_image_id);if(!s||s->telemetry.Empty())continue;const auto r=Summarize(p.ref_image_id,s->telemetry);
        csv<<r.id<<','<<r.valid<<','<<r.mean_err<<','<<r.under2<<','<<r.under10<<','<<r.candidate_same<<','<<r.anchor_same<<','<<r.plane_depth<<','<<r.plane_normal<<','<<r.radius_violation<<'\n';
        if(!first)json<<",\n";first=false;
        json<<"    {\"id\":"<<r.id<<",\"valid\":"<<r.valid<<",\"mean_depth_error_m\":"<<r.mean_err<<",\"within_2cm\":"<<r.under2<<",\"within_10cm\":"<<r.under10<<",\"candidate_same_surface_ratio\":"<<r.candidate_same<<",\"anchor_same_surface_ratio\":"<<r.anchor_same<<",\"plane_depth_error_m\":"<<r.plane_depth<<",\"plane_normal_error_deg\":"<<r.plane_normal<<",\"radius_violation\":"<<r.radius_violation<<"}";
    }
    json<<"\n  ]\n}\n";
}

void ArtifactWriter::WriteCaseManifest(const std::string& case_name,
                                       const std::string& description,
                                       const std::vector<Problem>& problems,
                                       Scene& scene,
                                       const boost::filesystem::path& point_cloud,
                                       const std::string& status,
                                       GroundTruthProvider* ground_truth) {
    const auto case_dir = root_ / case_name; boost::filesystem::create_directories(case_dir);
    std::ofstream out((case_dir / "manifest.json").string());
    out << "{\n  \"schema\":2,\n  \"name\":\"" << Escape(case_name) << "\",\n"
        << "  \"description\":\"" << Escape(description) << "\",\n  \"status\":\"" << Escape(status) << "\",\n"
        << "  \"dense_root\":\"" << Escape(dense_.string()) << "\",\n"
        << "  \"point_cloud\":\"" << Escape(boost::filesystem::relative(point_cloud, case_dir).generic_string()) << "\",\n"
        << "  \"telemetry\":" << (ground_truth?"true":"false") << ",\n";
    if(ground_truth) out << "  \"ground_truth_point_cloud\":\"../_ground_truth/aligned_scan_preview.ply\",\n";
    if(ground_truth){const auto& g=ground_truth->Config();out<<"  \"ground_truth\":{\"scan_ply\":\""<<Escape(g.scan_ply.string())<<"\",\"scan_alignment_mlp\":\""<<Escape(g.scan_alignment_mlp.string())<<"\",\"global_points\":"<<ground_truth->GlobalPointCount()<<"},\n";}
    if (ground_truth)
        out << "  \"layers\":[\"rgb\",\"depth\",\"normal\",\"state\",\"fine_edge\",\"coarse_region\",\"texture_complexity\",\"gt_depth\",\"gt_normal\",\"gt_geometry_edge\",\"edge_relation\",\"gt_surface_label\",\"es_candidate_count\",\"es_same_ratio\",\"candidate_count\",\"candidate_same_ratio\",\"anchor_same_ratio\",\"plane_depth_error\",\"plane_normal_error\",\"radius_violation\",\"final_depth_error\",\"matching_cost\",\"adaptive_radius\"],\n";
    else
        out << "  \"layers\":[\"rgb\",\"depth\",\"normal\",\"state\"],\n";
    out << "  \"views\":[\n";
    for (size_t i = 0; i < problems.size(); ++i) {
        const auto& p = problems[i]; const cv::Mat& img = scene.GrayImage(p.ref_image_id); Camera cam = scene.ScaledCamera(p.ref_image_id, img.cols, img.rows);
        out << "    {\"id\":" << p.ref_image_id << ",\"image\":\"images/" << FormatIndex(p.ref_image_id) << ".jpg\""
            << ",\"artifact\":\"views/" << FormatIndex(p.ref_image_id) << "\""
            << ",\"width\":" << cam.width << ",\"height\":" << cam.height << ",\"depth_min\":" << cam.depth_min << ",\"depth_max\":" << cam.depth_max;
        out << ",\"K\":"; WriteFloatArray(out, cam.K, 9); out << ",\"R\":"; WriteFloatArray(out, cam.R, 9); out << ",\"t\":"; WriteFloatArray(out, cam.t, 3); out << ",\"c\":"; WriteFloatArray(out, cam.c, 3);
        out << ",\"sources\":["; for (size_t j = 0; j < p.src_image_ids.size(); ++j) { if (j) out << ','; out << p.src_image_ids[j]; } out << "]}" << (i + 1 == problems.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
}

}  // namespace dpe
