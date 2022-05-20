#include "Utils.h"

#include <fstream>
#include <sstream>

std::string Utils::readFileToString(const std::string& aFileName)
{
    std::ifstream kernelFile(aFileName);
    std::stringstream srcStream;
    srcStream << kernelFile.rdbuf();
    return srcStream.str();
}
