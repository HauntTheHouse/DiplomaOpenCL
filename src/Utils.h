#pragma once

#include <string>

namespace Utils
{
    std::string readFileToString(const std::string& aFileName);
    std::string selectFileInDirectory(const std::string& aDirectoryPath);

    template<typename T>
    void enterValue(T& choose, T leftLimit, T rightLimit);
}
