//
// Created by magnus-desktop on 9/28/25.
//
module;

#include <string>
#include <iomanip>

export module Pale.Utils.StringFormatting;

export namespace Pale::Utils {
    std::string formatBytes(std::size_t bytes) {
        constexpr double KB = 1024.0;
        constexpr double MB = 1024.0 * KB;
        constexpr double GB = 1024.0 * MB;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        if (bytes >= GB) oss << (bytes / GB) << " GB";
        else if (bytes >= MB) oss << (bytes / MB) << " MB";
        else if (bytes >= KB) oss << (bytes / KB) << " KB";
        else oss << bytes << " B";
        return oss.str();
    }
}
