#include <iostream>
#include <string>

#include <CL/opencl.hpp>

#include "SparseMatrix.h"
#include "Algorithms.h"

int main()
{
    std::string matrixFileName = "sparse_matrix_112.txt";
    //std::string matrixFileName = "bcsstm12.mtx";
    SparseMatrix sparseMatrix("data/" + matrixFileName);
    sparseMatrix.fillVectorBWithValue(200.0);

    {
        std::ofstream fileStream("processed_" + matrixFileName);
        sparseMatrix.print(fileStream);
    }

    //const auto result = Algorithms::conjugateGradientCpu(sparseMatrix);
    const auto result = Algorithms::conjugateGradientGpu(sparseMatrix);

    for (const auto val : result.x)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Iterations: " << result.iterationNum << std::endl;
    std::cout << "Residual length: " << result.residualLength << std::endl << std::endl;
    std::cout << "Compute time: " << result.computeTime.value << " " << Time::toString(result.computeTime.measure) << std::endl;
}
