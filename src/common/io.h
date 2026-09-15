#pragma once

#include "common/types.h"
#include <opencv2/opencv.hpp>

namespace dpe {

std::string FormatIndex(int index);
bool ReadCamera(const boost::filesystem::path& path, Camera& camera);
bool ReadBinMat(const boost::filesystem::path& path, cv::Mat& mat);
bool WriteBinMat(const boost::filesystem::path& path, const cv::Mat& mat);
void WriteDepthPreview(const boost::filesystem::path& path, const cv::Mat& depth,
                       float depth_min, float depth_max);
void WriteNormalPreview(const boost::filesystem::path& path, const cv::Mat& normal);
void WriteReliabilityPreview(const boost::filesystem::path& path, const cv::Mat& reliability);

}  // namespace dpe
