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
    std::cout << "\n\nChoose method that solves SLAE:\n";
    std::cout << "1. Conjugate gradient method on GPU (matrix dimension must be less than work group size of GPU)\n";
    std::cout << "2. Conjugate gradient method on GPU\n";
    std::cout << "3. Conjugate gradient method on CPU\n";
    std::cout << "4. Conjugate gradient method on GPU and CPU\n";
    std::cout << "5. Conjugate gradient method using ViennaCL library (no preconditioners)\n";
    std::cout << "6. Conjugate gradient method using ViennaCL library (Jacobi preconditioner)\n";
    std::cout << "7. Steepest descent method on GPU\n";
    std::cout << "8. Steepest descent method on CPU\n";

    const int chosenMethod = Utils::selectOption(1, 7);

    Algorithms::Result result;
    if (chosenMethod == 1)
        result = Algorithms::conjugateGradientGpu(sparseMatrix);
    else if (chosenMethod == 2)
        result = Algorithms::conjugateGradientGpuScaled(sparseMatrix, false);
    else if (chosenMethod == 3)
        result = Algorithms::conjugateGradientCpu(sparseMatrix);
    else if (chosenMethod == 4)
        result = Algorithms::conjugateGradientGpuScaled(sparseMatrix, true);
    else if (chosenMethod == 5)
        result = Algorithms::conjugateGradientViennaCL(sparseMatrix, false);
    else if (chosenMethod == 6)
        result = Algorithms::conjugateGradientViennaCL(sparseMatrix, true);
    else if (chosenMethod == 7)
        result = Algorithms::steepestDescentGpu(sparseMatrix);
    else if (chosenMethod == 8)
        result = Algorithms::steepestDescentCpu(sparseMatrix);


    // Here are results of computations:
    //    vector x (that had to be calculated in Ax = b),
    //    number of iterations,
    //    residual length�
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
    std::cout << *result.timeInfo;

    return 0;
}
