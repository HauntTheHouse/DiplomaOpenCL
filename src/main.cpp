#include <iostream>
#include <string>

#include "SparseMatrix.h"
#include "Algorithms.h"
#include "Utils.h"

int main()
{
    // Here is the choice of a sparse matrix (A) for solving SLAE (Ax = b). The data with matrices is in the data/ directory
    std::cout << "Choose an available matrix to solve:" << std::endl;

    std::string dataDir = "data/";
    const int numFiles = Utils::getNumFilesInDirectory(std::filesystem::directory_iterator(dataDir));
    Utils::printFilesInDirectory(std::filesystem::directory_iterator(dataDir));

    const int id = Utils::selectOption(1, numFiles);
    const auto chosenMatrixFile = Utils::getFileNameByIdInDirectory(std::filesystem::directory_iterator(dataDir), id);

    SparseMatrix sparseMatrix(chosenMatrixFile);
    std::cout << "\nDimention of matrix : " << sparseMatrix.getDimension() << 'x' << sparseMatrix.getDimension() << std::endl;
    std::cout << "Number of non-zero values: " << sparseMatrix.getValuesNum() << std::endl;


    // Here is the choice of how to fill the vector b in Ax = b
    std::cout << "\n\nFill vector b:" << std::endl;
    std::cout << "1. With random values" << std::endl;
    std::cout << "2. With concrete value" << std::endl;

    const int chosenFillVector = Utils::selectOption(1, 2);

    if (chosenFillVector == 1)
    {
        sparseMatrix.fillVectorBWithRandomValues(-1'000'000, 1'000'000);
    }
    else if (chosenFillVector == 2)
    {
        std::cout << "\nEnter value: ";
        double value;
        Utils::enterValue(value);
        sparseMatrix.fillVectorBWithValue(value);
    }


    // Here is the choice of method that will be used for solving SLAE using GPU or CPU
    std::cout << "\n\nChoose method that solves SLAE:" << std::endl;
    std::cout << "1. Conjugate gradient method on GPU" << std::endl;
    std::cout << "2. Conjugate gradient method on GPU (scaled for big matrices)" << std::endl;
    std::cout << "3. Conjugate gradient method on CPU" << std::endl;
    std::cout << "4. Steepest descent method on GPU" << std::endl;
    std::cout << "5. Steepest descent method on CPU" << std::endl;

    const int chosenMethod = Utils::selectOption(1, 5);

    Algorithms::Result result;
    if (chosenMethod == 1)
        result = Algorithms::conjugateGradientGpu(sparseMatrix);
    else if (chosenMethod == 2)
        result = Algorithms::conjugateGradientGpuScaled(sparseMatrix);
    else if (chosenMethod == 3)
        result = Algorithms::conjugateGradientCpu(sparseMatrix);
    else if (chosenMethod == 4)
        result = Algorithms::steepestDescentGpu(sparseMatrix);
    else if (chosenMethod == 5)
        result = Algorithms::steepestDescentCpu(sparseMatrix);


    // Here are results of computations:
    //    vector x (that had to be calculated in Ax = b),
    //    number of iterations,
    //    residual lengthá
    //    compute time
    std::cout << "\n\nResults:\n" << std::endl;
    std::cout << "x = [ ";
    for (const auto val : result.x)
    {
        std::cout << val << ", ";
    }
    std::cout << "]\n\n";

    const auto b = Algorithms::matrixVectorMultiplication(sparseMatrix, result.x);
    std::cout << "\nAx = [ ";
    for (const auto val : b)
    {
        std::cout << val << ", ";
    }
    std::cout << "]\n\n";

    std::cout << "Iterations: " << result.iterationNum << std::endl;
    std::cout << "Residual length: " << result.residualLength << std::endl << std::endl;
    std::cout << "Compute time: " << result.computedTime.value << " " << Timer::toString(result.computedTime.measure) << std::endl;
    if (result.trueComputedTime.has_value())
    {
        std::cout << "Compute time without reading and writing buffers: " << result.trueComputedTime.value().value << " " << Timer::toString(result.trueComputedTime.value().measure) << std::endl;
    }

    return 0;
}
