#pragma once

#include <vector>
#include <memory>
#include <optional>

#include "TimeInfo.h"
#include "Timer.h"

class SparseMatrix;

namespace Algorithms
{
	struct Result
	{
		std::vector<double> x;
		int iterationNum;
		double residualLength;

		std::unique_ptr<TimeInfo> timeInfo;
	};

	Result conjugateGradientGpu(const SparseMatrix& aSparseMatrix);
	Result conjugateGradientGpuScaled(const SparseMatrix& aSparseMatrix);
	Result conjugateGradientCpu(const SparseMatrix& aSparseMatrix);

	Result steepestDescentGpu(const SparseMatrix& aSparseMatrix);
	Result steepestDescentCpu(const SparseMatrix& aSparseMatrix);

	std::vector<double> matrixVectorMultiplication(const SparseMatrix& aSparseMatrix, const std::vector<double>& aVector);
}
