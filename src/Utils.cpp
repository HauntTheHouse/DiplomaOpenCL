#include "Utils.h"

#include <fstream>
#include <sstream>
#include <iostream>

namespace Utils
{

std::string readFileToString(const std::string& aFileName)
{
    std::ifstream kernelFile(aFileName);
    std::stringstream srcStream;
    srcStream << kernelFile.rdbuf();
    return srcStream.str();
}

int getNumFilesInDirectory(const std::filesystem::directory_iterator& aDirectory)
{
    int i = 0;
    for (const auto& file : aDirectory) ++i;

    return i;
}

void printFilesInDirectory(const std::filesystem::directory_iterator& aDirectory)
{
    int i = 1;
    for (const auto& file : aDirectory)
    {
        const auto path = file.path().string();
        std::cout << i++ << ". " << path.substr(path.find_last_of('/') + 1) << '\n';
    }
}

std::string getFileNameByIdInDirectory(const std::filesystem::directory_iterator& aDirectory, int aId)
{
    int i = 1;
    for (const auto& file : aDirectory)
    {
        if (i++ == aId)
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

int selectOption(int aLeftLimit, int aRightLimit)
{
    std::cout << "\nSelect option: ";
    int chosenOption;
    Utils::enterValue(chosenOption, aLeftLimit, aRightLimit);
    return chosenOption;
}

}
