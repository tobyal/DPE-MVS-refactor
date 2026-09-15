#pragma once

#include "common/types.h"
#include "preprocessing/edge_detection.h"

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_map>

namespace dpe {

struct SceneView {
    int ref_id = -1;
    std::vector<int> image_ids;   // ref first
    std::vector<const cv::Mat*> images_float;
    std::vector<Camera> cameras;
    int width = 0;
    int height = 0;
    int scale_size = 1;
};

class Scene {
public:
    explicit Scene(const boost::filesystem::path& dense_folder);

    const boost::filesystem::path& Root() const { return root_; }
    const cv::Mat& GrayImage(int image_id);
    const cv::Mat& ColorImage(int image_id);
    const cv::Mat& ScaledGrayFloat(int image_id, int scale_size);
    const cv::Mat& ScaledGrayU8(int image_id, int scale_size);
    Camera ScaledCamera(int image_id, int target_width, int target_height);

    // Pyramid-sized images and guidance are retained only for the active level.
    // Raw images/cameras remain resident for the whole reconstruction.
    void SetActiveScale(int scale_size);
    int ActiveScale() const { return active_scale_size_; }

    SceneView MakeView(const Problem& problem);
    const EdgeGuidanceHost& Guidance(int image_id, int scale_size, const DPEParams& params);

    std::vector<Problem> LoadProblems(const boost::filesystem::path& output_folder) const;
    int ComputePyramidLevels(const std::vector<Problem>& problems);
    bool CheckImageSizes(const std::vector<Problem>& problems);

private:
    struct ScaleKey {
        int image = -1;
        int scale = 1;
        bool operator==(const ScaleKey& rhs) const { return image == rhs.image && scale == rhs.scale; }
    };
    struct ScaleKeyHash {
        size_t operator()(const ScaleKey& k) const {
            return (static_cast<size_t>(k.image) << 16) ^ static_cast<size_t>(k.scale);
        }
    };

    boost::filesystem::path root_;
    boost::filesystem::path image_dir_;
    boost::filesystem::path camera_dir_;

    int active_scale_size_ = 0;
    std::unordered_map<int, cv::Mat> gray_images_;
    std::unordered_map<int, cv::Mat> color_images_;
    std::unordered_map<int, Camera> cameras_;

    // Active-level working cache: bounded to one pyramid level.
    std::unordered_map<int, cv::Mat> scaled_float_;
    std::unordered_map<int, cv::Mat> scaled_u8_;
    std::unordered_map<int, EdgeGuidanceHost> guidance_;

    // High-resolution edge crossing always references the coarsest fine edge.
    // Keep just that small map per image across levels.
    std::unordered_map<ScaleKey, cv::Mat, ScaleKeyHash> lowres_edges_;
};

}  // namespace dpe
