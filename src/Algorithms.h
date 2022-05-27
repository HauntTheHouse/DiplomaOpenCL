#pragma once

#include <vector>

#include "Time.h"

class SparseMatrix;

namespace Algorithms
{
	struct Result
	{
		std::vector<double> x;
		int iterationNum;
		double residualLength;
		Time::ComputedTime computeTime;
	};

	Result conjugateGradientCpu(const SparseMatrix& aSparseMatrix);
	Result conjugateGradientGpu(const SparseMatrix& aSparseMatrix);
	Result conjugateGradientGpuScaled(const SparseMatrix& aSparseMatrix);

	Result steepestDescentCpu(const SparseMatrix& aSparseMatrix);
	Result steepestDescentGpu(const SparseMatrix& aSparseMatrix);
}
