#include "scene/scene.h"
#include "common/io.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace dpe {

Scene::Scene(const boost::filesystem::path& dense_folder)
    : root_(dense_folder), image_dir_(root_ / "images"), camera_dir_(root_ / "cams") {}

const cv::Mat& Scene::GrayImage(int image_id) {
    auto it = gray_images_.find(image_id);
    if (it != gray_images_.end()) return it->second;
    const auto path = image_dir_ / (FormatIndex(image_id) + ".jpg");
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("Cannot read image: " + path.string());
    return gray_images_.emplace(image_id, std::move(img)).first->second;
}

const cv::Mat& Scene::ColorImage(int image_id) {
    auto it = color_images_.find(image_id);
    if (it != color_images_.end()) return it->second;
    const auto path = image_dir_ / (FormatIndex(image_id) + ".jpg");
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("Cannot read image: " + path.string());
    return color_images_.emplace(image_id, std::move(img)).first->second;
}

void Scene::SetActiveScale(int scale_size) {
    scale_size = std::max(1, scale_size);
    if (active_scale_size_ == scale_size) return;
    scaled_float_.clear();
    scaled_u8_.clear();
    guidance_.clear();
    active_scale_size_ = scale_size;
}

const cv::Mat& Scene::ScaledGrayFloat(int image_id, int scale_size) {
    if (active_scale_size_ != scale_size) SetActiveScale(scale_size);
    auto it = scaled_float_.find(image_id);
    if (it != scaled_float_.end()) return it->second;
    const cv::Mat& src_u8 = GrayImage(image_id);
    cv::Mat src, dst;
    src_u8.convertTo(src, CV_32F);
    const int w = std::max(1, static_cast<int>(std::round(src.cols / static_cast<double>(scale_size))));
    const int h = std::max(1, static_cast<int>(std::round(src.rows / static_cast<double>(scale_size))));
    if (w == src.cols && h == src.rows) dst = src;
    else cv::resize(src, dst, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    return scaled_float_.emplace(image_id, std::move(dst)).first->second;
}

const cv::Mat& Scene::ScaledGrayU8(int image_id, int scale_size) {
    if (active_scale_size_ != scale_size) SetActiveScale(scale_size);
    auto it = scaled_u8_.find(image_id);
    if (it != scaled_u8_.end()) return it->second;
    cv::Mat dst;
    ScaledGrayFloat(image_id, scale_size).convertTo(dst, CV_8U);
    return scaled_u8_.emplace(image_id, std::move(dst)).first->second;
}

Camera Scene::ScaledCamera(int image_id, int target_width, int target_height) {
    auto it = cameras_.find(image_id);
    if (it == cameras_.end()) {
        Camera c;
        const auto path = camera_dir_ / (FormatIndex(image_id) + "_cam.txt");
        if (!ReadCamera(path, c)) throw std::runtime_error("Cannot read camera: " + path.string());
        const cv::Mat& img = GrayImage(image_id);
        c.width = img.cols;
        c.height = img.rows;
        it = cameras_.emplace(image_id, c).first;
    }
    Camera c = it->second;
    const float sx = target_width / static_cast<float>(std::max(1, c.width));
    const float sy = target_height / static_cast<float>(std::max(1, c.height));
    c.K[0] *= sx; c.K[2] *= sx;
    c.K[4] *= sy; c.K[5] *= sy;
    c.width = target_width;
    c.height = target_height;
    return c;
}

SceneView Scene::MakeView(const Problem& problem) {
    SceneView view;
    view.ref_id = problem.ref_image_id;
    view.scale_size = problem.scale_size;
    view.image_ids.push_back(problem.ref_image_id);
    for (int id : problem.src_image_ids) {
        if (static_cast<int>(view.image_ids.size()) >= problem.params.num_images) break;
        view.image_ids.push_back(id);
    }
    if (view.image_ids.size() < 2) throw std::runtime_error("Not enough source views");
    const cv::Mat& ref = ScaledGrayFloat(problem.ref_image_id, problem.scale_size);
    view.width = ref.cols;
    view.height = ref.rows;
    for (int id : view.image_ids) {
        const cv::Mat& img = ScaledGrayFloat(id, problem.scale_size);
        view.images_float.push_back(&img);
        view.cameras.push_back(ScaledCamera(id, img.cols, img.rows));
    }
    return view;
}

const EdgeGuidanceHost& Scene::Guidance(int image_id, int scale_size, const DPEParams& params) {
    if (active_scale_size_ != scale_size) SetActiveScale(scale_size);
    auto it = guidance_.find(image_id);
    if (it != guidance_.end()) return it->second;

    int scale_log2 = 0;
    while ((1 << scale_log2) < scale_size) ++scale_log2;
    EdgeGuidanceHost g = BuildEdgeGuidance(GrayImage(image_id), ScaledGrayU8(image_id, scale_size),
                                           scale_log2, params);

    // On high-resolution inputs the edge-crossing test uses the coarsest
    // fine-edge map at every level. Cache that compact map independently so
    // changing pyramid levels does not keep every scaled image resident.
    if (params.high_res_img) {
        const int edge_scale = std::max(1, params.max_scale_size);
        const ScaleKey low_key{image_id, edge_scale};
        auto low_it = lowres_edges_.find(low_key);
        if (low_it == lowres_edges_.end()) {
            const cv::Mat& raw = GrayImage(image_id);
            cv::Mat raw_f, scaled_f, scaled_u8;
            raw.convertTo(raw_f, CV_32F);
            const int w = std::max(1, static_cast<int>(std::round(raw.cols / static_cast<double>(edge_scale))));
            const int h = std::max(1, static_cast<int>(std::round(raw.rows / static_cast<double>(edge_scale))));
            if (w == raw.cols && h == raw.rows) scaled_f = raw_f;
            else cv::resize(raw_f, scaled_f, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
            scaled_f.convertTo(scaled_u8, CV_8U);
            cv::Mat edge = ComputeFineEdges(scaled_u8);
            low_it = lowres_edges_.emplace(low_key, std::move(edge)).first;
        }
        g.lowres_edges = low_it->second;
    } else {
        g.lowres_edges = g.fine_edges;
    }
    return guidance_.emplace(image_id, std::move(g)).first->second;
}

std::vector<Problem> Scene::LoadProblems(const boost::filesystem::path& output_folder) const {
    std::ifstream file((root_ / "pair.txt").string());
    if (!file) throw std::runtime_error("Cannot open pair.txt");
    int count = 0;
    file >> count;
    std::vector<Problem> problems;
    problems.reserve(count);
    for (int i = 0; i < count; ++i) {
        Problem p;
        p.index = i;
        file >> p.ref_image_id;
        int nsrc = 0;
        file >> nsrc;
        for (int j = 0; j < nsrc; ++j) {
            int id = -1;
            float score = 0.0f;
            file >> id >> score;
            if (score > 0.0f) p.src_image_ids.push_back(id);
        }
        p.dense_folder = root_;
        p.result_folder = output_folder / "views" / FormatIndex(p.ref_image_id);
        boost::filesystem::create_directories(p.result_folder);
        problems.push_back(std::move(p));
    }
    return problems;
}

int Scene::ComputePyramidLevels(const std::vector<Problem>& problems) {
    if (problems.empty()) return 0;
    const cv::Mat& img = GrayImage(problems.front().ref_image_id);
    int max_size = std::max(img.cols, img.rows);
    int levels = 1;
    while (max_size > 800) { max_size /= 2; ++levels; }
    return levels;
}

bool Scene::CheckImageSizes(const std::vector<Problem>& problems) {
    if (problems.empty()) return false;
    const cv::Size size = GrayImage(problems.front().ref_image_id).size();
    for (const auto& p : problems)
        if (GrayImage(p.ref_image_id).size() != size) return false;
    return true;
}

}  // namespace dpe
