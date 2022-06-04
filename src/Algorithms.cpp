#include "Algorithms.h"

#include <iostream>
#include <string>
#include <CL/opencl.hpp>

#include "SparseMatrix.h"
#include "Utils.h"

namespace Algorithms
{

namespace
{
    struct OpenCLParameters
    {
        cl::Context context;
        cl::Program program;
        cl::Device device;
    };

    OpenCLParameters initOpenCL(const std::string& aPathToKernel)
    {
        cl::Context context;
        cl::Program program;
        cl::Device device;

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
        {
            std::cerr << "\n\nNo platforms found!" << std::endl;
            exit(-1);
        }

        std::cout << "\n\nChoose an available platform:" << std::endl;
        for (size_t i = 0; i < platforms.size(); ++i)
        {
            std::cout << i + 1 << ". " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;

            std::vector<cl::Device> devices;
            platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

            std::cout << "\tDevices:" << std::endl;
            for (const auto& dev : devices)
            {
                std::cout << '\t' << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
                std::cout << "\tMAX_WORK_GROUP_SIZE: " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
                std::cout << "\tMAX_COMPUTE_UNITS: " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            }
        }

        std::cout << "\nSelect option: ";
        int chosenPlatform;
        Utils::enterValue(chosenPlatform, 1, (int)platforms.size());
        auto& platform = platforms[chosenPlatform - 1];
        std::cout << std::endl << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        device = devices.back();
        std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        std::string src = Utils::readFileToString(aPathToKernel);

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
            exit(-2);
        }

        std::cout << "\nSolving of SLAE..." << std::endl;
        return { context, program, device };
    }
}

Result conjugateGradientGpu(const SparseMatrix& aSparseMatrix)
{
    const int dimension = aSparseMatrix.getDimension();
    const int numValues = aSparseMatrix.getValuesNum();

    std::vector<double> x(dimension);
    std::vector<double> result(2);

    const auto [context, program, device] = initOpenCL("kernels/conjugateGradient.cl");

    cl::Kernel kernel(program, "conjugateGradient");

    int deviceMaxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    if (dimension > deviceMaxWorkGroupSize)
    {
        std::cerr << "Dimension of matrix is bigger than max work group size of the device. Use scaledConjugateGradientGpu() algorithm instead" << std::endl;
        exit(-3);
    }

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

    cl::CommandQueue queue(context, device);

    auto computeLinearSystem = [&]()
    {
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(dimension), cl::NDRange(dimension));

        queue.enqueueReadBuffer(xBuf, CL_TRUE, 0, x.size() * sizeof(double), x.data());
        queue.enqueueReadBuffer(resultBuf, CL_TRUE, 0, result.size() * sizeof(double), result.data());
    };

    const auto measuredTime = Timer::computeTime(computeLinearSystem);

    return { x, static_cast<int>(result[0]), result[1], measuredTime };
}

Result conjugateGradientGpuScaled(const SparseMatrix& aSparseMatrix)
{
    const auto [context, program, device] = initOpenCL("kernels/conjugateGradientScaled.cl");

    cl::Kernel init_kernel(program, "init");
    cl::Kernel update_r_length_old_kernel(program, "update_r_length");
    cl::Kernel update_A_times_p_kernel(program, "update_A_times_p");
    cl::Kernel calculate_alpha_kernel(program, "calculate_alpha");
    cl::Kernel update_guess_kernel(program, "update_guess");
    cl::Kernel update_r_length_new_kernel(program, "update_r_length");
    cl::Kernel update_direction_kernel(program, "update_direction");
    cl::Kernel sync_r_dot_r_kernel(program, "sync_r_dot_r");

    cl::CommandQueue queue = cl::CommandQueue(context, device);

    int deviceMaxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    const int dimension = aSparseMatrix.getDimension();
    const int numValues = aSparseMatrix.getValuesNum();

    int local = dimension;
    int global = dimension;
    if (dimension > deviceMaxWorkGroupSize)
    {
        local = deviceMaxWorkGroupSize;
        global += (local - dimension % local);
    }

    std::vector<int> startIds(dimension);
    std::vector<int> endIds(dimension);
    std::vector<double> r(dimension);
    std::vector<double> p(dimension);

    double old_r_dot_r, new_r_dot_r, r_length;

    std::vector<double> A_times_p(dimension);

    std::vector<double> x(dimension);
    std::vector<double> result(2);

    double alpha;
    int iterations = 0;

    cl::Buffer startIdBuf(context, CL_MEM_READ_WRITE, dimension * sizeof(int));
    cl::Buffer endIdBuf(context, CL_MEM_READ_WRITE, dimension * sizeof(int));

    cl::Buffer xBuf(context, CL_MEM_READ_WRITE, dimension * sizeof(double));
    cl::Buffer rBuf(context, CL_MEM_READ_WRITE, dimension * sizeof(double));
    cl::Buffer pBuf(context, CL_MEM_READ_WRITE, dimension * sizeof(double));

    cl::Buffer rowsBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(int), const_cast<int*>(aSparseMatrix.getRowIds()));
    cl::Buffer colsBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(int), const_cast<int*>(aSparseMatrix.getColIds()));
    cl::Buffer valuesBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(double), const_cast<double*>(aSparseMatrix.getValues()));
    cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, dimension * sizeof(double), const_cast<double*>(aSparseMatrix.getVectorB()));
    cl::Buffer resultBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 2 * sizeof(double));

    cl::Buffer old_r_dot_r_Buf(context, CL_MEM_READ_WRITE, sizeof(double));
    cl::Buffer new_r_dot_r_Buf(context, CL_MEM_READ_WRITE, sizeof(double));
    cl::Buffer r_length_Buf(context, CL_MEM_READ_WRITE, sizeof(double));
    cl::Buffer A_times_p_Buf(context, CL_MEM_READ_WRITE, dimension * sizeof(double));
    cl::Buffer alpha_Buf(context, CL_MEM_READ_WRITE, sizeof(double));

    init_kernel.setArg(0, sizeof(int), &numValues);
    init_kernel.setArg(1, startIdBuf);
    init_kernel.setArg(2, endIdBuf);
    init_kernel.setArg(3, xBuf);
    init_kernel.setArg(4, rBuf);
    init_kernel.setArg(5, pBuf);
    init_kernel.setArg(6, rowsBuf);
    init_kernel.setArg(7, bBuf);

    update_r_length_old_kernel.setArg(0, sizeof(int), &dimension);
    update_r_length_old_kernel.setArg(1, rBuf);
    update_r_length_old_kernel.setArg(2, old_r_dot_r_Buf);
    update_r_length_old_kernel.setArg(3, r_length_Buf);

    update_A_times_p_kernel.setArg(0, A_times_p_Buf);
    update_A_times_p_kernel.setArg(1, startIdBuf);
    update_A_times_p_kernel.setArg(2, endIdBuf);
    update_A_times_p_kernel.setArg(3, valuesBuf);
    update_A_times_p_kernel.setArg(4, pBuf);
    update_A_times_p_kernel.setArg(5, colsBuf);

    calculate_alpha_kernel.setArg(0, sizeof(int), &dimension);
    calculate_alpha_kernel.setArg(1, old_r_dot_r_Buf);
    calculate_alpha_kernel.setArg(2, A_times_p_Buf);
    calculate_alpha_kernel.setArg(3, pBuf);
    calculate_alpha_kernel.setArg(4, alpha_Buf);

    update_guess_kernel.setArg(0, xBuf);
    update_guess_kernel.setArg(1, rBuf);
    update_guess_kernel.setArg(2, alpha_Buf);
    update_guess_kernel.setArg(3, pBuf);
    update_guess_kernel.setArg(4, A_times_p_Buf);

    update_r_length_new_kernel.setArg(0, sizeof(int), &dimension);
    update_r_length_new_kernel.setArg(1, rBuf);
    update_r_length_new_kernel.setArg(2, new_r_dot_r_Buf);
    update_r_length_new_kernel.setArg(3, r_length_Buf);

    update_direction_kernel.setArg(0, old_r_dot_r_Buf);
    update_direction_kernel.setArg(1, new_r_dot_r_Buf);
    update_direction_kernel.setArg(2, rBuf);
    update_direction_kernel.setArg(3, pBuf);

    sync_r_dot_r_kernel.setArg(0, old_r_dot_r_Buf);
    sync_r_dot_r_kernel.setArg(1, new_r_dot_r_Buf);

    auto computeLinearSystem = [&]()
    {
        queue.enqueueNDRangeKernel(init_kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local));

        queue.enqueueNDRangeKernel(update_r_length_old_kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
        queue.enqueueReadBuffer(r_length_Buf, CL_TRUE, 0, sizeof(double), &r_length);

        while (iterations < 50000 && r_length >= 0.01)
        {
            queue.enqueueNDRangeKernel(update_A_times_p_kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local));
            queue.enqueueNDRangeKernel(calculate_alpha_kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
            queue.enqueueNDRangeKernel(update_guess_kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local));
            queue.enqueueNDRangeKernel(update_r_length_new_kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
            queue.enqueueNDRangeKernel(update_direction_kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local));
            queue.enqueueNDRangeKernel(sync_r_dot_r_kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
            queue.enqueueReadBuffer(r_length_Buf, CL_TRUE, 0, sizeof(double), &r_length);
            iterations++;
        }

        queue.enqueueReadBuffer(xBuf, CL_TRUE, 0, x.size() * sizeof(double), x.data());
    };

    const auto measuredTime = Timer::computeTime(computeLinearSystem);

    return { x, iterations, r_length, measuredTime };
}

Result steepestDescentGpu(const SparseMatrix& aSparseMatrix)
{
    const int dimension = aSparseMatrix.getDimension();
    const int numValues = aSparseMatrix.getValuesNum();

    std::vector<double> x(dimension);
    std::vector<double> result(2);

    const auto [context, program, device] = initOpenCL("kernels/steepestDescent.cl");

    cl::Kernel kernel(program, "steepestDescent");

    int deviceMaxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    std::cout << "DEVICE_MAX_WORK_GROUP_SIZE: " << deviceMaxWorkGroupSize << std::endl << std::endl;
    if (dimension > deviceMaxWorkGroupSize)
    {
        std::cerr << "Dimension of matrix is bigger than max work group size of the device" << std::endl;
        exit(-3);
    }

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

    cl::CommandQueue queue(context, device);

    auto computeLinearSystem = [&]()
    {
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(dimension), cl::NDRange(dimension));

        queue.enqueueReadBuffer(xBuf, CL_TRUE, 0, x.size() * sizeof(double), x.data());
        queue.enqueueReadBuffer(resultBuf, CL_TRUE, 0, result.size() * sizeof(double), result.data());
    };

    const auto measuredTime = Timer::computeTime(computeLinearSystem);

    return { x, static_cast<int>(result[0]), result[1], measuredTime };
}

Result conjugateGradientCpu(const SparseMatrix& aSparseMatrix)
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

    const auto measuredTime = Timer::computeTime(computeLinearSystem);

    return { x, iterations, residualLength, measuredTime };
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

    const auto measuredTime = Timer::computeTime(computeLinearSystem);

    return { x, iterations, residualLength, measuredTime };
}

}
