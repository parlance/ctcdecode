#include <cxxopts.hpp>

#include "build_fst.h"

int main(int argc, char* argv[])
{
    cxxopts::Options options("build_fst", "A program to build FST from the lexicon files");

    std::vector<std::string> lexicon_paths;

    options.add_options()(
        "v,vocab-path", "Path to file containing labels", cxxopts::value<std::string>())(
        "i,lexicon-paths",
        "Path to lexicon files. Multiple paths can be provided. Each line in the file should be "
        "like this: <frequency> <word> <tokens seperated by space> ",
        cxxopts::value<std::vector<std::string>>(lexicon_paths))(
        "o,output-path",
        "Path to a output file. Both optimized and unoptimized files gets created. optimized file "
        "contains `.opt` extension",
        cxxopts::value<std::string>())(
        "freq-threshold",
        "Words having frequency greater than or equal to this threshold will be considered "
        "(Default = -1 i.e all are considered in this case) ",
        cxxopts::value<int>()->default_value("-1"))(
        "fst-path",
        "Path to a fst file. Default is empty. If provided, the words will be added on top of this "
        "FST. (NOTE: Unoptimized fst file need to be provided in this case, otherwise the "
        "generated FST will "
        "not be proper ) ",
        cxxopts::value<std::string>()->default_value(""))("h,help", "Print usage");

    options.parse_positional({ "vocab-path", "lexicon-paths" });

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (!lexicon_paths.empty()) {
        std::cout << lexicon_paths.size() << " lexicon paths are provided: " << std::endl;
        for (auto path : lexicon_paths) {
            std::cout << path << std::endl;
        }
    } else {
        std::cout << "No lexicon paths are provided. Exiting" << std::endl;
        exit(0);
    }

    std::string vocab_path = result["vocab-path"].as<std::string>();
    std::string output_path = result["output-path"].as<std::string>();
    std::string fst_path = result["fst-path"].as<std::string>();
    int freq_threshold = result["freq-threshold"].as<int>();

    std::cout << "Freq threshold: " << freq_threshold << std::endl;

    construct_fst(vocab_path, lexicon_paths, fst_path, output_path, freq_threshold, true);

    return 0;
}
