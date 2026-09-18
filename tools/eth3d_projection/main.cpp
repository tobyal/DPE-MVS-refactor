#include "common/io.h"
#include "evaluation_types.h"
#include "ply_reader.h"
#include "projection.h"

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Options {
    boost::filesystem::path dense_folder;
    boost::filesystem::path output;
    boost::filesystem::path accuracy_2cm;
    boost::filesystem::path accuracy_10cm;
    boost::filesystem::path completeness_2cm;
    boost::filesystem::path completeness_10cm;
    int radius = 2;
    double alpha = 0.7;
    int threads = 1;
};

struct ViewData {
    int image_id = -1;
    cv::Mat image;
    dpe::Camera camera;
};

struct CloudJob {
    boost::filesystem::path input;
    boost::filesystem::path output_directory;
    dpe::eth3d::EvaluationKind kind = dpe::eth3d::EvaluationKind::Accuracy;
    double tolerance_meters = 0.0;
    std::string name;
};

struct ViewStats {
    std::size_t projected_points = 0;
    std::size_t visible_pixels = 0;
};

void PrintUsage(const char* executable) {
    std::cerr
        << "Usage: " << executable << " --dense-folder <path> --output <path>\n"
        << "  --accuracy-2cm <accuracy.tolerance_0.02.ply>\n"
        << "  --accuracy-10cm <accuracy.tolerance_0.1.ply>\n"
        << "  --completeness-2cm <completeness.tolerance_0.02.ply>\n"
        << "  --completeness-10cm <completeness.tolerance_0.1.ply>\n"
        << "  [--radius 2] [--alpha 0.7] [--threads 4]\n";
}

std::string OptionValue(int argc, char** argv, int* index, const std::string& argument) {
    const std::size_t equals = argument.find('=');
    if (equals != std::string::npos) return argument.substr(equals + 1);
    if (*index + 1 >= argc) throw std::runtime_error(argument + " requires a value");
    return argv[++(*index)];
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    options.threads = std::max(1, std::min(4, static_cast<int>(hardware_threads == 0 ? 1 : hardware_threads)));

    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        const std::string name = argument.substr(0, argument.find('='));
        if (name == "--dense-folder") {
            options.dense_folder = OptionValue(argc, argv, &index, argument);
        } else if (name == "--output") {
            options.output = OptionValue(argc, argv, &index, argument);
        } else if (name == "--accuracy-2cm") {
            options.accuracy_2cm = OptionValue(argc, argv, &index, argument);
        } else if (name == "--accuracy-10cm") {
            options.accuracy_10cm = OptionValue(argc, argv, &index, argument);
        } else if (name == "--completeness-2cm") {
            options.completeness_2cm = OptionValue(argc, argv, &index, argument);
        } else if (name == "--completeness-10cm") {
            options.completeness_10cm = OptionValue(argc, argv, &index, argument);
        } else if (name == "--radius") {
            options.radius = std::stoi(OptionValue(argc, argv, &index, argument));
        } else if (name == "--alpha") {
            options.alpha = std::stod(OptionValue(argc, argv, &index, argument));
        } else if (name == "--threads") {
            options.threads = std::stoi(OptionValue(argc, argv, &index, argument));
        } else if (argument == "--help" || argument == "-h") {
            PrintUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("Unknown option: " + argument);
        }
    }

    if (options.dense_folder.empty() || options.output.empty() ||
        options.accuracy_2cm.empty() || options.accuracy_10cm.empty() ||
        options.completeness_2cm.empty() || options.completeness_10cm.empty()) {
        throw std::runtime_error("Dense folder, output, and all four evaluation PLY paths are required");
    }
    if (options.radius < 0) throw std::runtime_error("--radius must be non-negative");
    if (options.alpha < 0.0 || options.alpha > 1.0) {
        throw std::runtime_error("--alpha must be in [0, 1]");
    }
    if (options.threads < 1) throw std::runtime_error("--threads must be at least 1");
    return options;
}

std::vector<int> ReadReferenceImageIds(const boost::filesystem::path& pair_path) {
    std::ifstream input(pair_path.string());
    if (!input) throw std::runtime_error("Cannot open pair file: " + pair_path.string());

    int count = 0;
    input >> count;
    if (!input || count <= 0) throw std::runtime_error("Invalid pair.txt reference count");

    std::vector<int> image_ids;
    std::set<int> seen;
    image_ids.reserve(count);
    for (int problem = 0; problem < count; ++problem) {
        int reference_id = -1;
        int source_count = 0;
        input >> reference_id >> source_count;
        if (!input || reference_id < 0 || source_count < 0) {
            throw std::runtime_error("Malformed pair.txt problem entry");
        }
        for (int source = 0; source < source_count; ++source) {
            int source_id = -1;
            float score = 0.0f;
            input >> source_id >> score;
            if (!input) throw std::runtime_error("Malformed pair.txt source entry");
        }
        if (seen.insert(reference_id).second) image_ids.push_back(reference_id);
    }
    return image_ids;
}

std::vector<ViewData> LoadViews(const boost::filesystem::path& dense_folder) {
    const std::vector<int> image_ids = ReadReferenceImageIds(dense_folder / "pair.txt");
    std::vector<ViewData> views;
    views.reserve(image_ids.size());
    for (int image_id : image_ids) {
        const std::string index = dpe::FormatIndex(image_id);
        const boost::filesystem::path image_path = dense_folder / "images" / (index + ".jpg");
        const boost::filesystem::path camera_path = dense_folder / "cams" / (index + "_cam.txt");

        ViewData view;
        view.image_id = image_id;
        view.image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (view.image.empty()) throw std::runtime_error("Cannot read image: " + image_path.string());
        if (!dpe::ReadCamera(camera_path, view.camera)) {
            throw std::runtime_error("Cannot read camera: " + camera_path.string());
        }
        view.camera.width = view.image.cols;
        view.camera.height = view.image.rows;
        views.push_back(std::move(view));
    }
    return views;
}

void RunCloudJob(const CloudJob& job,
                 const std::vector<ViewData>& views,
                 const Options& options) {
    std::cout << "Loading " << job.name << ": " << job.input.string() << '\n';
    const std::vector<dpe::eth3d::EvaluationPoint> points =
        dpe::eth3d::ReadEvaluationPly(job.input);
    std::cout << "  points: " << points.size() << '\n';
    boost::filesystem::create_directories(job.output_directory);

    std::vector<ViewStats> stats(views.size());
    std::atomic<std::size_t> next_view(0);
    std::mutex error_mutex;
    std::exception_ptr error;
    const int thread_count = std::min(options.threads, static_cast<int>(views.size()));
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    for (int thread = 0; thread < thread_count; ++thread) {
        workers.emplace_back([&]() {
            while (true) {
                const std::size_t view_index = next_view.fetch_add(1);
                if (view_index >= views.size()) return;
                {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (error) return;
                }
                try {
                    const ViewData& view = views[view_index];
                    dpe::eth3d::ProjectionResult projection =
                        dpe::eth3d::ProjectEvaluationCloud(
                            points, view.camera, view.image, job.kind, job.tolerance_meters,
                            options.radius, options.alpha);
                    const boost::filesystem::path output_path =
                        job.output_directory / (dpe::FormatIndex(view.image_id) + ".png");
                    if (!cv::imwrite(output_path.string(), projection.image)) {
                        throw std::runtime_error("Cannot write output image: " + output_path.string());
                    }
                    stats[view_index].projected_points = projection.projected_points;
                    stats[view_index].visible_pixels = projection.visible_pixels;
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!error) error = std::current_exception();
                    return;
                }
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    if (error) std::rethrow_exception(error);

    for (std::size_t index = 0; index < views.size(); ++index) {
        std::cout << "  " << dpe::FormatIndex(views[index].image_id)
                  << ": projected_points=" << stats[index].projected_points
                  << ", visible_pixels=" << stats[index].visible_pixels << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        const Options options = ParseOptions(argc, argv);
        if (!boost::filesystem::exists(options.dense_folder / "pair.txt")) {
            throw std::runtime_error("Dense folder must contain pair.txt");
        }

        const std::vector<ViewData> views = LoadViews(options.dense_folder);
        std::cout << "Reference views: " << views.size() << '\n'
                  << "Splat radius: " << options.radius << " px, alpha: " << options.alpha
                  << ", threads: " << options.threads << '\n';

        const std::vector<CloudJob> jobs = {
            {options.accuracy_2cm, options.output / "accuracy_2cm",
             dpe::eth3d::EvaluationKind::Accuracy, 0.02, "accuracy_2cm"},
            {options.accuracy_10cm, options.output / "accuracy_10cm",
             dpe::eth3d::EvaluationKind::Accuracy, 0.10, "accuracy_10cm"},
            {options.completeness_2cm, options.output / "completeness_2cm",
             dpe::eth3d::EvaluationKind::Completeness, 0.02, "completeness_2cm"},
            {options.completeness_10cm, options.output / "completeness_10cm",
             dpe::eth3d::EvaluationKind::Completeness, 0.10, "completeness_10cm"},
        };
        for (const CloudJob& job : jobs) RunCloudJob(job, views, options);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "eth3d_projection failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
