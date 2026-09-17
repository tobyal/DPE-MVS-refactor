#include "scene/scene.h"
#include "scene/reconstruction_state.h"
#include "runtime/cuda_context.h"
#include "pipeline/dpe_pipeline.h"
#include "fusion/fusion.h"
#include "diagnostics/file_diagnostic_sink.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <memory>

namespace {

struct Options {
    int gpu_index = 0;
    bool debug = false;
    bool diagnostics = false;
    bool diagnostics_all_views = false;
    int diagnostic_view = -1;
    boost::filesystem::path output;
    boost::filesystem::path diagnostics_output;
};

Options ParseOptions(int argc, char** argv) {
    Options options;
    bool gpu_set = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--debug" || arg == "--vis=all") {
            options.debug = true;
        } else if (arg.rfind("--output=", 0) == 0) {
            options.output = arg.substr(9);
        } else if (arg == "--output") {
            if (i + 1 >= argc) throw std::runtime_error("--output requires a path");
            options.output = argv[++i];
        } else if (arg == "--diagnostics") {
            options.diagnostics = true;
        } else if (arg.rfind("--diagnostics=", 0) == 0) {
            options.diagnostics = true;
            options.diagnostics_output = arg.substr(14);
        } else if (arg == "--diagnostic-view=all") {
            options.diagnostics = true;
            options.diagnostics_all_views = true;
        } else if (arg.rfind("--diagnostic-view=", 0) == 0) {
            options.diagnostics = true;
            options.diagnostic_view = std::stoi(arg.substr(18));
        } else if (arg.rfind("--vis=", 0) == 0) {
            // `none` and `final` remain accepted for compatibility. `final`
            // currently writes the final reconstruction state only through fusion.
            const std::string value = arg.substr(6);
            if (value != "none" && value != "final")
                throw std::runtime_error("invalid --vis value: " + value);
        } else if (arg.rfind("--checkpoint=", 0) == 0 || arg.rfind("--profile=", 0) == 0) {
            // Accepted so existing DPE-MVS-fast launch scripts keep working.
        } else if (!gpu_set && !arg.empty() && arg[0] != '-') {
            options.gpu_index = std::atoi(arg.c_str());
            gpu_set = true;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: DPE <dense_folder> [gpu_index] [--output=<folder>] "
                     "[--debug] [--diagnostics[=<folder>]]\n";
        return EXIT_FAILURE;
    }

    try {
        const boost::filesystem::path input(argv[1]);
        const Options options = ParseOptions(argc, argv);
        const boost::filesystem::path output = options.output.empty() ? input / "DPE" : options.output;

        if (!boost::filesystem::exists(input / "images") ||
            !boost::filesystem::exists(input / "cams") ||
            !boost::filesystem::exists(input / "pair.txt")) {
            throw std::runtime_error("input folder must contain images/, cams/, and pair.txt");
        }
        boost::filesystem::create_directories(output / "views");

        dpe::Scene scene(input);
        auto problems = scene.LoadProblems(output);
        for (auto& p : problems) p.show_medium_result = options.debug;

        dpe::ReconstructionState reconstruction;
        dpe::CudaContext cuda(options.gpu_index);
        std::unique_ptr<dpe::FileDiagnosticSink> diagnostics;
        if (options.diagnostics) {
            const boost::filesystem::path diagnostics_root = options.diagnostics_output.empty()
                ? output / "analysis"
                : options.diagnostics_output;
            const int diagnostic_view = options.diagnostic_view >= 0
                ? options.diagnostic_view
                : problems.front().ref_image_id;
            diagnostics.reset(new dpe::FileDiagnosticSink(
                diagnostics_root, diagnostic_view, options.diagnostics_all_views));
            std::cout << "Diagnostics: " << diagnostics_root.string() << '\n';
            std::cout << "Stage trace view: "
                      << (options.diagnostics_all_views
                          ? std::string("all")
                          : std::to_string(diagnostic_view)) << '\n';
        }
        dpe::DPEPipeline pipeline(scene, reconstruction, cuda, diagnostics.get());
        pipeline.Run(problems);
        dpe::RunFusion(scene, reconstruction, problems, output / "DPE.ply", diagnostics.get());

        std::cout << "DPE.ply: " << (output / "DPE.ply").string() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "DPE failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
