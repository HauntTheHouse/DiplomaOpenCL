#include "Algorithms.h"

#include <iostream>
#include <CL/opencl.hpp>
#include "SparseMatrix.h"
#include "Utils.h"

namespace Algorithms
{
    ResultConjugateGradient Algorithms::conjugateGradientCpu(const SparseMatrix& aSparseMatrix)
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

    ResultConjugateGradient Algorithms::conjugateGradientGpu(const SparseMatrix& aSparseMatrix)
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
            for (const auto &dev : devices)
            {
                std::cout << '\t' << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
                std::cout << "\tMAX_WORK_GROUP_SIZE: " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
                for (auto &itemSize : dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>())
                {
                    std::cout << "\t\tMAX_ITEM_SIZE: " << itemSize << std::endl;
                }
                std::cout << "\tMAX_WORK_ITEM_DIMENSIONS: " << dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
                std::cout << "\tMAX_COMPUTE_UNITS: " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            }
        }

    //    auto& platform = platforms.front();
        auto& platform = platforms.back();
        std::cout << std::endl << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        device = devices.back();
        std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        std::string method = "conjugateGradient";

        std::cout << "Algorithm that solves linear equation: " << method << std::endl;
        std::string src = Utils::readFileToString("kernels/" + method + ".cl");

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

        const auto numValues = aSparseMatrix.getValuesNum();
        const auto dimension = aSparseMatrix.getDimension();

        std::vector<double> x(dimension);
        std::vector<double> result(2);

        cl::Buffer rowsBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(int), const_cast<int*>(aSparseMatrix.getRowIds()));
        cl::Buffer colsBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(int), const_cast<int*>(aSparseMatrix.getColIds()));
        cl::Buffer valuesBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(double), const_cast<double*>(aSparseMatrix.getValues()));
        cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, dimension * sizeof(double), const_cast<double*>(aSparseMatrix.getVectorB()));
        cl::Buffer xBuf(context, CL_MEM_READ_WRITE, dimension * sizeof(double));
        cl::Buffer resultBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 2 * sizeof(double));

        cl::Kernel kernel(program, method.c_str());

        int kernelWorkGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        int deviceMaxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        std::cout << "KERNEL_WORK_GROUP_SIZE: " << kernelWorkGroupSize << std::endl;
        std::cout << "DEVICE_MAX_WORK_GROUP_SIZE: " << deviceMaxWorkGroupSize << std::endl;

        if (dimension > kernelWorkGroupSize)
        {
            std::cerr << "Dimention of the matrix is bigger then kernel work group size of your device" << std::endl;
            exit(-2);
        }

        cl::CommandQueue queue(context, device);

        auto computeLinearSystem = [&]()
        {
            kernel.setArg(0, sizeof(int), &dimension);
            kernel.setArg(1, sizeof(int), &numValues);
            kernel.setArg(2, cl::Local(dimension * sizeof(double)));
            kernel.setArg(3, xBuf);
            kernel.setArg(4, cl::Local(dimension * sizeof(double)));
            kernel.setArg(5, cl::Local(dimension * sizeof(double)));
            kernel.setArg(6, rowsBuf);
            kernel.setArg(7, colsBuf);
            kernel.setArg(8, valuesBuf);
            kernel.setArg(9, bBuf);
            kernel.setArg(10, resultBuf);

            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(dimension), cl::NDRange(dimension));

            queue.enqueueReadBuffer(xBuf, CL_TRUE, 0, x.size() * sizeof(double), x.data());
            queue.enqueueReadBuffer(resultBuf, CL_TRUE, 0, result.size() * sizeof(double), result.data());
        };

        const auto measuredTime = Time::compute(computeLinearSystem);

        return { x, static_cast<int>(result[0]), result[1], measuredTime };
    }
}
