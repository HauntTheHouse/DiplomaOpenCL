#include <iostream>
#include <string>

#include <CL/opencl.hpp>

#include "SparseMatrix.h"
#include "Algorithms.h"
#include "Utils.h"

int main()
{
    std::cout << "Choose an available matrix to solve:" << std::endl;
    const auto chosenMatrixFile = Utils::selectFileInDirectory("data/");
    SparseMatrix sparseMatrix(chosenMatrixFile);
    std::cout << "\nDimention of matrix : " << sparseMatrix.getDimension() << 'x' << sparseMatrix.getDimension() << std::endl;
    std::cout << "Number of non-zero values: " << sparseMatrix.getValuesNum() << std::endl;

    std::cout << "\n\nFill vector b:" << std::endl;
    std::cout << "1. With random values" << std::endl;
    std::cout << "2. With concrete values" << std::endl;

    std::cout << "\nSelect option: ";
    int chosenFillVector;
    Utils::enterValue(chosenFillVector, 1, 2);

    if (chosenFillVector == 1)
    {
        sparseMatrix.fillVectorBWithRandomValues(-1'000'000, 1'000'000);
    }
    else if (chosenFillVector == 2)
    {
        std::cout << "\nEnter value: ";
        double value;
        Utils::enterValue(value, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
        sparseMatrix.fillVectorBWithValue(value);
    }

    std::cout << "\n\nChoose method that solves SLAE:" << std::endl;
    std::cout << "1. Conjugate gradient method on GPU" << std::endl;
    std::cout << "2. Conjugate gradient method on GPU (scaled for big matrices)" << std::endl;
    std::cout << "3. Conjugate gradient method on CPU" << std::endl;
    std::cout << "4. Steepest descent method on GPU" << std::endl;
    std::cout << "5. Steepest descent method on CPU" << std::endl;

    std::cout << "\nSelect option: ";
    int chosenMethod;
    Utils::enterValue(chosenMethod, 1, 5);
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

    std::cout << "\n\nResults:\n" << std::endl;
    std::cout << "x = [ ";
    for (const auto val : result.x)
    {
        std::cout << val << ", ";
    }
    std::cout << "]\n\n";

    std::cout << "Iterations: " << result.iterationNum << std::endl;
    std::cout << "Residual length: " << result.residualLength << std::endl << std::endl;
    std::cout << "Compute time: " << result.computedTime.value << " " << Timer::toString(result.computedTime.measure) << std::endl;
}
