#pragma once

#include <vector>

#include "Time.h"

class SparseMatrix;

namespace Algorithms
{
	struct ResultConjugateGradient
	{
		std::vector<double> x;
		int iterationNum;
		double residualLength;
		Time::ComputedTime computeTime;
	};

	ResultConjugateGradient conjugateGradientCpu(const SparseMatrix& aSparseMatrix);
	ResultConjugateGradient conjugateGradientGpu(const SparseMatrix& aSparseMatrix);
}
