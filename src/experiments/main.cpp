#include "scene/scene.h"
#include "runtime/cuda_context.h"
#include "experiments/experiment_runner.h"
#include "experiments/experiment_suite.h"
#include "experiments/ground_truth/ground_truth_provider.h"

#include <boost/filesystem.hpp>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: DPEExperiment <dense_folder> [gpu_index] [--output=<experiment_root>] "
                     "[--gt-scan=/path/scan.ply] [--gt-mlp=/path/scan_alignment.mlp] "
                     "[--gt-splat-radius=1] [--telemetry-all]\n";
        return EXIT_FAILURE;
    }
    try {
        boost::filesystem::path dense(argv[1]);
        int gpu = 0;
        boost::filesystem::path output = dense / "DPE_experiments";
        dpe::GroundTruthConfig gt_cfg;
        bool gt_requested = false;
        bool telemetry_all = false;
        for (int i = 2; i < argc; ++i) {
            std::string a(argv[i]);
            if (a.rfind("--output=", 0) == 0) output = a.substr(9);
            else if (a.rfind("--gt-scan=",0)==0) { gt_cfg.scan_ply=a.substr(10); gt_requested=true; }
            else if (a.rfind("--gt-mlp=",0)==0) { gt_cfg.scan_alignment_mlp=a.substr(9); gt_requested=true; }
            else if (a.rfind("--gt-splat-radius=",0)==0) gt_cfg.splat_radius=std::max(0,std::atoi(a.substr(18).c_str()));
            else if (a == "--telemetry-all") telemetry_all = true;
            else if (!a.empty() && a[0] != '-') gpu = std::atoi(a.c_str());
            else throw std::runtime_error("unknown option: " + a);
        }
        if (gt_requested && gt_cfg.scan_alignment_mlp.empty() && gt_cfg.scan_ply.empty())
            throw std::runtime_error("GT requested but neither --gt-scan nor --gt-mlp was supplied");

        dpe::Scene scene(dense);
        auto base = scene.LoadProblems(output / "_template");
        dpe::CudaContext cuda(gpu);
        std::unique_ptr<dpe::GroundTruthProvider> gt;
        if (gt_requested) {
            gt.reset(new dpe::GroundTruthProvider(gt_cfg));
            std::cout << "ETH3D GT loaded: " << gt->GlobalPointCount() << " aligned scan points\n";
        }
        dpe::ExperimentRunner runner(scene, cuda, output, gt.get());
        auto suite = dpe::MakeDefaultExperimentSuite();
        if (telemetry_all) for (auto& c : suite) c.telemetry = true;
        runner.Run(base, suite);
        std::cout << "experiment root: " << output.string() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "DPEExperiment failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
