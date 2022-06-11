#pragma once

#include <vector>
#include <optional>
#include "Timer.h"

class SparseMatrix;

namespace Algorithms
{
	struct Result
	{
		std::vector<double> x;
		int iterationNum;
		double residualLength;
		Timer::ComputedTime computedTime;
		std::optional<Timer::ComputedTime> trueComputedTime;
	};

	Result conjugateGradientGpu(const SparseMatrix& aSparseMatrix);
	Result conjugateGradientGpuScaled(const SparseMatrix& aSparseMatrix);
	Result conjugateGradientCpu(const SparseMatrix& aSparseMatrix);

	Result steepestDescentGpu(const SparseMatrix& aSparseMatrix);
	Result steepestDescentCpu(const SparseMatrix& aSparseMatrix);

	std::vector<double> matrixVectorMultiplication(const SparseMatrix& aSparseMatrix, const std::vector<double>& aVector);
}
