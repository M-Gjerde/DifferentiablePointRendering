#pragma once
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace viewer {
    struct DatasetEntry {
        std::filesystem::path directory;
        std::optional<std::filesystem::path> groundTruth;
        std::string label;
    };
    inline bool isDatasetDirectory(const std::filesystem::path& directory) {
        std::error_code error;
        return std::filesystem::is_directory(directory/"images",error) &&
            std::filesystem::is_regular_file(directory/"scene.xml",error) &&
            std::filesystem::is_regular_file(directory/"points.ply",error);
    }
    inline std::optional<std::filesystem::path> datasetGroundTruth(const std::filesystem::path& directory) {
        std::error_code error;
        const auto path = directory / "gt.ply";
        if (std::filesystem::is_regular_file(path,error)) return path;
        return {};
    }
    inline DatasetEntry datasetEntry(std::filesystem::path directory) {
        directory=directory.lexically_normal();
        if(directory.filename().empty()&&directory!=directory.parent_path())directory=directory.parent_path();
        auto label=directory.filename().string();
        if(label.ends_with("_pbdr"))label.resize(label.size()-5);
        std::replace(label.begin(),label.end(),'_',' ');
        if(!label.empty())label[0]=static_cast<char>(std::toupper(static_cast<unsigned char>(label[0])));
        return {directory,datasetGroundTruth(directory),label};
    }
    inline std::vector<DatasetEntry> scanDatasets(const std::filesystem::path& root) {
        std::vector<DatasetEntry> result;
        std::error_code error;
        if (!std::filesystem::is_directory(root,error)) return result;
        for (const auto& item : std::filesystem::directory_iterator(root,error)) {
            if (error) break;
            const auto directory=item.path();
            if (!isDatasetDirectory(directory)) continue;
            result.push_back(datasetEntry(directory));
        }
        std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.label<b.label;});
        return result;
    }
}
