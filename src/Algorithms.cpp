#include "Algorithms.h"

#include <iostream>
#include <string>
#include <CL/opencl.hpp>

#include "SparseMatrix.h"
#include "Utils.h"
#include "Time.h"

namespace Algorithms
{
    namespace
    {
        Time::ComputedTime computeWithOpenCL(const std::string& aKernelPath, const std::function<void(const cl::Context&, cl::Kernel&, cl::CommandQueue&)>& aProgramEnqueue)
        {
            cl::Context context;
            cl::Program program;
            cl::Device device;

            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if (platforms.empty())
            {
                std::cerr << "No platforms found!" << std::endl;
                exit(1);
            }

            for (int i = 0; i < platforms.size(); ++i)
            {
                std::cout << i + 1 << ". " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;

                std::vector<cl::Device> devices;
                platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

                std::cout << "\tDevices:" << std::endl;
                for (const auto& dev : devices)
                {
                    std::cout << '\t' << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
                    std::cout << "\tMAX_WORK_GROUP_SIZE: " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
                    for (auto& itemSize : dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>())
                    {
                        std::cout << "\t\tMAX_ITEM_SIZE: " << itemSize << std::endl;
                    }
                    std::cout << "\tMAX_WORK_ITEM_DIMENSIONS: " << dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
                    std::cout << "\tMAX_COMPUTE_UNITS: " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
                }
            }

            auto& platform = platforms.front();
            //auto& platform = platforms.back();
            std::cout << std::endl << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            device = devices.back();
            std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

            auto first = aKernelPath.find_first_of('/') + 1;
            auto last = aKernelPath.find_last_of('.');
            std::string method = aKernelPath.substr(first, last - first);
            std::cout << "Algorithm that solves linear equation: " << method << std::endl;

            std::string src = Utils::readFileToString(aKernelPath);

            cl::Program::Sources sources;
            sources.push_back(src);

            context = cl::Context(device);
            program = cl::Program(context, sources);

            auto err = program.build();
            std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            if (err != CL_BUILD_SUCCESS)
            {
                std::cerr << "Error!" << std::endl;
                std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) << std::endl;
                std::cerr << "Build Log:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
                exit(-1);
            }

            cl::Kernel kernel(program, method.c_str());

            int kernelWorkGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
            int deviceMaxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
            std::cout << "KERNEL_WORK_GROUP_SIZE: " << kernelWorkGroupSize << std::endl;
            std::cout << "DEVICE_MAX_WORK_GROUP_SIZE: " << deviceMaxWorkGroupSize << std::endl;

            cl::CommandQueue queue(context, device);

            const auto measuredTime = Time::compute(aProgramEnqueue, context, kernel, queue);
            return measuredTime;
        }
    }

    Result Algorithms::conjugateGradientCpu(const SparseMatrix& aSparseMatrix)
    {
        std::vector<double> x(aSparseMatrix.getDimension());
        int iterations;
        double residualLength;

        auto computeLinearSystem = [
                dim = aSparseMatrix.getDimension(),
                num_vals = aSparseMatrix.getValuesNum(),
                rows = aSparseMatrix.getRowIds(),
                cols = aSparseMatrix.getColIds(),
                A = aSparseMatrix.getValues(),
                b = aSparseMatrix.getVectorB(),
                &x = x,
                &iterations = iterations,
                &residualLength = residualLength]()
            {
                std::vector<double> r(dim);
                std::vector<double> A_times_p(dim);
                std::vector<double> p(dim);
                double alpha, r_length, old_r_dot_r, new_r_dot_r;
                double Ap_dot_p;

                for (int i = 0; i < dim; ++i)
                {
                    x[i] = 0.0;
                    r[i] = b[i];
                    p[i] = b[i];
                }

                old_r_dot_r = 0.0;
                for (int i = 0; i < dim; i++)
                {
                    old_r_dot_r += r[i] * r[i];
                }
                r_length = sqrt(old_r_dot_r);

                int iteration = 0;
                while (iteration < 50000 && r_length >= 0.01)
                {
                    int etalon = 0;
                    int j = 0;

                    for (int i = 0; i < dim; ++i)
                    {
                        A_times_p[i] = 0.0;
                        while (etalon == rows[j])
                        {
                            A_times_p[i] += A[j] * p[cols[j]];
                            j++;
                        }
                        etalon++;
                    }

                    Ap_dot_p = 0.0;
                    for (int i = 0; i < dim; i++)
                    {
                        Ap_dot_p += A_times_p[i] * p[i];
                    }
                    alpha = old_r_dot_r / Ap_dot_p;

                    for (int i = 0; i < dim; i++)
                    {
                        x[i] += alpha * p[i];
                        r[i] -= alpha * A_times_p[i];
                    }

                    new_r_dot_r = 0.0;
                    for (int i = 0; i < dim; i++)
                    {
                        new_r_dot_r += r[i] * r[i];
                    }
                    r_length = sqrt(new_r_dot_r);

                    for (int i = 0; i < dim; i++)
                    {
                        p[i] = r[i] + (new_r_dot_r / old_r_dot_r) * p[i];
                    }

                    old_r_dot_r = new_r_dot_r;
                    iteration++;
                }
                iterations = iteration;
                residualLength = r_length;
            };

            const auto measuredTime = Time::compute(computeLinearSystem);

            return { x, iterations, residualLength, measuredTime };
    }

    Result Algorithms::conjugateGradientGpu(const SparseMatrix& aSparseMatrix)
    {
        const auto dimension = aSparseMatrix.getDimension();
        const auto numValues = aSparseMatrix.getValuesNum();

        std::vector<double> x(dimension);
        std::vector<double> result(2);

        auto computeLinearSystem = [&](const cl::Context& context, cl::Kernel& kernel, cl::CommandQueue& queue)
        {
            cl::Buffer rowsBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(int), const_cast<int*>(aSparseMatrix.getRowIds()));
            cl::Buffer colsBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(int), const_cast<int*>(aSparseMatrix.getColIds()));
            cl::Buffer valuesBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(double), const_cast<double*>(aSparseMatrix.getValues()));
            cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, dimension * sizeof(double), const_cast<double*>(aSparseMatrix.getVectorB()));
            cl::Buffer xBuf(context, CL_MEM_READ_WRITE, dimension * sizeof(double));
            cl::Buffer resultBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 2 * sizeof(double));

            kernel.setArg(0, sizeof(int), &dimension);
            kernel.setArg(1, sizeof(int), &numValues);
            kernel.setArg(2, cl::Local(dimension * sizeof(double)));
            kernel.setArg(3, cl::Local(dimension * sizeof(double)));
            kernel.setArg(4, cl::Local(dimension * sizeof(double)));
            kernel.setArg(5, rowsBuf);
            kernel.setArg(6, colsBuf);
            kernel.setArg(7, valuesBuf);
            kernel.setArg(8, bBuf);
            kernel.setArg(9, xBuf);
            kernel.setArg(10, resultBuf);

            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(dimension), cl::NDRange(dimension));

            queue.enqueueReadBuffer(xBuf, CL_TRUE, 0, x.size() * sizeof(double), x.data());
            queue.enqueueReadBuffer(resultBuf, CL_TRUE, 0, result.size() * sizeof(double), result.data());
        };

        const auto measuredTime = computeWithOpenCL("kernels/conjugateGradient.cl", computeLinearSystem);

        return { x, static_cast<int>(result[0]), result[1], measuredTime };
    }

    Result steepestDescentCpu(const SparseMatrix& aSparseMatrix)
    {
        std::vector<double> x(aSparseMatrix.getDimension());
        int iterations;
        double residualLength;

        auto computeLinearSystem = [
                dim = aSparseMatrix.getDimension(),
                num_vals = aSparseMatrix.getValuesNum(),
                rows = aSparseMatrix.getRowIds(),
                cols = aSparseMatrix.getColIds(),
                A = aSparseMatrix.getValues(),
                b = aSparseMatrix.getVectorB(),
                &x = x,
                &iterations = iterations,
                &residualLength = residualLength]()
            {
                std::vector<double> r(dim);
                std::vector<double> A_times_r(dim);

                double alpha, r_length;
                double r_dot_r, Ar_dot_r;

                for (int i = 0; i < dim; ++i)
                {
                    r[i] = b[i];
                    x[i] = 0.0;
                }

                int iteration = 0;
                r_length = 0.01;
                while (iteration < 50000 && r_length >= 0.01)
                {
                    int etalon = 0;
                    int j = 0;

                    for (int i = 0; i < dim; ++i)
                    {
                        A_times_r[i] = 0.0;
                        while (etalon == rows[j])
                        {
                            A_times_r[i] += A[j] * r[cols[j]];
                            j++;
                        }
                        etalon++;
                    }

                    r_dot_r = 0.0;
                    Ar_dot_r = 0.0;
                    for (int i = 0; i < dim; i++)
                    {
                        r_dot_r += r[i] * r[i];
                        Ar_dot_r += A_times_r[i] * r[i];
                    }
                    alpha = r_dot_r / Ar_dot_r;

                    for (int i = 0; i < dim; i++)
                    {
                        x[i] += alpha * r[i];
                        r[i] -= alpha * A_times_r[i];
                    }

                    r_length = sqrt(r_dot_r);
                    iteration++;
                }
                iterations = iteration;
                residualLength = r_length;
            };

            const auto measuredTime = Time::compute(computeLinearSystem);

            return { x, iterations, residualLength, measuredTime };
    }

    Result steepestDescentGpu(const SparseMatrix& aSparseMatrix)
    {
        const auto dimension = aSparseMatrix.getDimension();
        const auto numValues = aSparseMatrix.getValuesNum();

        std::vector<double> x(dimension);
        std::vector<double> result(2);

        auto computeLinearSystem = [&](const cl::Context& context, cl::Kernel& kernel, cl::CommandQueue& queue)
        {
            cl::Buffer rowsBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(int), const_cast<int*>(aSparseMatrix.getRowIds()));
            cl::Buffer colsBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(int), const_cast<int*>(aSparseMatrix.getColIds()));
            cl::Buffer valuesBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(double), const_cast<double*>(aSparseMatrix.getValues()));
            cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, dimension * sizeof(double), const_cast<double*>(aSparseMatrix.getVectorB()));
            cl::Buffer xBuf(context, CL_MEM_READ_WRITE, dimension * sizeof(double));
            cl::Buffer resultBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 2 * sizeof(double));

            kernel.setArg(0, sizeof(int), &dimension);
            kernel.setArg(1, sizeof(int), &numValues);
            kernel.setArg(2, cl::Local(dimension * sizeof(double)));
            kernel.setArg(3, cl::Local(dimension * sizeof(double)));
            kernel.setArg(4, rowsBuf);
            kernel.setArg(5, colsBuf);
            kernel.setArg(6, valuesBuf);
            kernel.setArg(7, bBuf);
            kernel.setArg(8, xBuf);
            kernel.setArg(9, resultBuf);

            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(dimension), cl::NDRange(dimension));

            queue.enqueueReadBuffer(xBuf, CL_TRUE, 0, x.size() * sizeof(double), x.data());
            queue.enqueueReadBuffer(resultBuf, CL_TRUE, 0, result.size() * sizeof(double), result.data());
        };

        const auto measuredTime = computeWithOpenCL("kernels/steepestDescent.cl", computeLinearSystem);

        return { x, static_cast<int>(result[0]), result[1], measuredTime };
    }
}
