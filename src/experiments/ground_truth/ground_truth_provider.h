#pragma once

#include "common/types.h"

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace dpe {

struct GroundTruthConfig {
    boost::filesystem::path scan_ply;
    boost::filesystem::path scan_alignment_mlp;
    int splat_radius = 1;
    float relative_depth_edge = 0.01f;
    float normal_edge_degrees = 20.0f;
    float surface_depth_ratio = 0.015f;
    float surface_normal_degrees = 25.0f;
};

struct GroundTruthFrame {
    cv::Mat depth;          // CV_32F, 0 for unavailable
    cv::Mat normal;         // CV_32FC3, world coordinates
    cv::Mat valid;          // CV_8U, 0/255
    cv::Mat geometry_edge;  // CV_8U, 0/255
    cv::Mat surface_label;  // CV_32S, 0 invalid, 1..N connected surface regions
};

class GroundTruthProvider {
public:
    explicit GroundTruthProvider(GroundTruthConfig config);

    const GroundTruthFrame& Frame(int image_id, const Camera& camera);
    const GroundTruthConfig& Config() const { return config_; }
    size_t GlobalPointCount() const { return points_.size(); }
    void WritePreviewPly(const boost::filesystem::path& path, size_t max_points = 1500000) const;

private:
    struct Point3 { float x, y, z; };
    struct Matrix44 { float v[16]; };
    struct CacheKey {
        int image_id = -1;
        int width = 0;
        int height = 0;
        bool operator==(const CacheKey& o) const { return image_id==o.image_id && width==o.width && height==o.height; }
    };

    void LoadAlignedScans();
    std::vector<Point3> ReadPlyVertices(const boost::filesystem::path& path) const;
    std::vector<std::pair<boost::filesystem::path, Matrix44>> ParseMlp() const;
    GroundTruthFrame Project(const Camera& camera) const;
    void EstimateNormalsAndSurfaces(const Camera& camera, GroundTruthFrame& frame) const;

    GroundTruthConfig config_;
    std::vector<Point3> points_;
    CacheKey last_key_;
    GroundTruthFrame last_frame_;
    bool has_last_ = false;
};

}  // namespace dpe
