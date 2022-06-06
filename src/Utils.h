#pragma once

#include <string>
#include <filesystem>

namespace Utils
{
    std::string readFileToString(const std::string& aFileName);

    int getNumFilesInDirectory(const std::filesystem::directory_iterator& aDirectory);
    void printFilesInDirectory(const std::filesystem::directory_iterator& aDirectory);
    std::string getFileNameByIdInDirectory(const std::filesystem::directory_iterator& aDirectory, int aId);

    template<typename T>
    void enterValue(T& aChoose, T aLeftLimit = -std::numeric_limits<T>::min(), T aRightLimit = std::numeric_limits<T>::max());
    int selectOption(int aLeftLimit = -std::numeric_limits<int>::min(), int aRightLimit= std::numeric_limits<int>::max());
}
