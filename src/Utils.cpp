#include "Utils.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

namespace Utils
{

std::string readFileToString(const std::string& aFileName)
{
    std::ifstream kernelFile(aFileName);
    std::stringstream srcStream;
    srcStream << kernelFile.rdbuf();
    return srcStream.str();
}

std::string selectFileInDirectory(const std::string& aDirectoryPath)
{
    int i = 1;
    for (const auto& file : std::filesystem::directory_iterator(aDirectoryPath))
    {
        const auto path = file.path().string();
        std::cout << i++ << ". " << path.substr(path.find_last_of('/') + 1) << '\n';
    }

    std::cout << "\nSelect option: ";
    int num;
    enterValue(num, 1, i - 1);

    i = 1;
    for (const auto& file : std::filesystem::directory_iterator(aDirectoryPath))
    {
        if (i++ == num)
            return file.path().string();
    }
}

template<typename T>
void enterValue(T& choose, T leftLimit, T rightLimit)
{
    while (true)
    {
        std::cin >> choose;
        std::cin.clear();
        std::cin.ignore();
        if (choose < leftLimit || choose > rightLimit)
            std::cout << "You enter incorrect value, try again...\n";
        else
            break;
    }
}
template void Utils::enterValue<int>(int&, int, int);
template void Utils::enterValue<double>(double&, double, double);

}
