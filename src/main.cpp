#include <iostream>
#include <string>

#include <CL/opencl.hpp>

#include "SparseMatrix.h"
#include "Algorithms.h"

int main()
{
    std::string matrixFileName = "sparse_matrix_112.txt";
    //std::string matrixFileName = "sparse_matrix_420.txt";
    //std::string matrixFileName = "bcsstk17.mtx";
    SparseMatrix sparseMatrix("data/" + matrixFileName);
    sparseMatrix.fillVectorBWithValue(200.0);

    {
        std::ofstream fileStream("processed_" + matrixFileName);
        sparseMatrix.print(fileStream);
    }

    //const auto result = Algorithms::conjugateGradientCpu(sparseMatrix);
    //const auto result = Algorithms::conjugateGradientGpu(sparseMatrix);
    const auto result = Algorithms::conjugateGradientGpuScaled(sparseMatrix);

    for (const auto val : result.x)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Iterations: " << result.iterationNum << std::endl;
    std::cout << "Residual length: " << result.residualLength << std::endl << std::endl;
    std::cout << "Compute time: " << result.computedTime.value << " " << Timer::toString(result.computedTime.measure) << std::endl;
}
