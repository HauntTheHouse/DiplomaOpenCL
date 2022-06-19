#include "Algorithms.h"

#include <iostream>
#include <string>
#include <cassert>
#include <chrono>
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
                std::cout << "\t\tMAX_WORK_GROUP_SIZE: " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
                std::cout << "\t\tMAX_COMPUTE_UNITS: " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            }
        }

        const int chosenPlatform = Utils::selectOption(1, platforms.size());

        auto& platform = platforms[chosenPlatform - 1];
        std::cout << std::endl << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        device = devices.front();
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

    cl_ulong getComputeTime(const cl::Event& event)
    {
        event.wait();
        return event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
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
        std::cerr << "\nDimension of matrix is bigger than max work group size of the device. Use scaledConjugateGradientGpu() algorithm instead" << std::endl;
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

    cl::CommandQueue queue(context, device, cl::QueueProperties::Profiling);


    cl_ulong kernel_compute_time = 0;
    cl_ulong read_buffer_time = 0;

    auto computeLinearSystem = [&]()
    {
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(dimension), cl::NDRange(dimension), nullptr, &event);
        kernel_compute_time += getComputeTime(event);

        queue.enqueueReadBuffer(xBuf, CL_TRUE, 0, x.size() * sizeof(double), x.data(), nullptr, &event);
        read_buffer_time += getComputeTime(event);
        queue.enqueueReadBuffer(resultBuf, CL_TRUE, 0, result.size() * sizeof(double), result.data(), nullptr, &event);
        read_buffer_time += getComputeTime(event);
    };

    auto timeInfo = std::make_unique<NonScaledTimeInfo>();
    timeInfo->total_compute_time  = Timer::computeTime(computeLinearSystem);
    timeInfo->total_kernel_time   = Timer::toAppropriateMeasure(kernel_compute_time + read_buffer_time);
    timeInfo->kernel_compute_time = Timer::toAppropriateMeasure(kernel_compute_time);
    timeInfo->read_buffer_time    = Timer::toAppropriateMeasure(read_buffer_time);

    return { x, static_cast<int>(result[0]), result[1], std::move(timeInfo) };
}

Result conjugateGradientGpuScaled(const SparseMatrix& aSparseMatrix, bool computeOneThreadedKernelsOnCPU)
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

    cl::CommandQueue queue = cl::CommandQueue(context, device, cl::QueueProperties::Profiling);

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

    cl_ulong init_time = 0;
    cl_ulong update_r_length_old_time = 0;
    cl_ulong update_A_times_p_time = 0;
    cl_ulong calc_alpha_time = 0;
    cl_ulong update_guess_time = 0;
    cl_ulong update_r_length_new_time = 0;
    cl_ulong update_direction_time = 0;
    cl_ulong sync_r_dot_r_time = 0;
    cl_ulong read_buffers_time = 0;
    cl_ulong total_kernel_time = 0;

    auto computeLinearSystem = [&]()
    {
        cl::Event event;
        queue.enqueueNDRangeKernel(init_kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local), nullptr, &event);
        init_time += getComputeTime(event);

        if (computeOneThreadedKernelsOnCPU)
        {
            queue.enqueueReadBuffer(rBuf, CL_TRUE, 0, r.size() * sizeof(double), r.data(), nullptr, &event);
            read_buffers_time += getComputeTime(event);

            const auto start = std::chrono::steady_clock::now();
                old_r_dot_r = 0.0;
                for (int i = 0; i < dimension; i++)
                {
                    old_r_dot_r += r[i] * r[i];
                }
                r_length = sqrt(old_r_dot_r);
            const auto end = std::chrono::steady_clock::now();

            update_r_length_old_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            queue.enqueueWriteBuffer(old_r_dot_r_Buf, CL_TRUE, 0, sizeof(double), &old_r_dot_r, nullptr, &event);
            read_buffers_time += getComputeTime(event);
        }
        else
        {
            queue.enqueueNDRangeKernel(update_r_length_old_kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1), nullptr, &event);
            update_r_length_old_time += getComputeTime(event);

            queue.enqueueReadBuffer(r_length_Buf, CL_TRUE, 0, sizeof(double), &r_length, nullptr, &event);
            read_buffers_time += getComputeTime(event);
        }

        while (iterations < 50000 && r_length >= 0.01)
        {
            queue.enqueueNDRangeKernel(update_A_times_p_kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local), nullptr, &event);
            update_A_times_p_time += getComputeTime(event);

            if (computeOneThreadedKernelsOnCPU)
            {
                queue.enqueueReadBuffer(A_times_p_Buf, CL_TRUE, 0, A_times_p.size() * sizeof(double), A_times_p.data(), nullptr, &event);
                read_buffers_time += getComputeTime(event);
                queue.enqueueReadBuffer(pBuf, CL_TRUE, 0, p.size() * sizeof(double), p.data(), nullptr, &event);
                read_buffers_time += getComputeTime(event);
                queue.enqueueReadBuffer(old_r_dot_r_Buf, CL_TRUE, 0, sizeof(double), &old_r_dot_r, nullptr, &event);
                read_buffers_time += getComputeTime(event);

                const auto start = std::chrono::steady_clock::now();
                    double Ap_dot_p = 0.0;
                    for (int i = 0; i < dimension; i++)
                    {
                        Ap_dot_p += A_times_p[i] * p[i];
                    }
                    alpha = old_r_dot_r / Ap_dot_p;
                const auto end = std::chrono::steady_clock::now();

                calc_alpha_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

                queue.enqueueWriteBuffer(alpha_Buf, CL_TRUE, 0, sizeof(double), &alpha, nullptr, &event);
                read_buffers_time += getComputeTime(event);
            }
            else
            {
                queue.enqueueNDRangeKernel(calculate_alpha_kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1), nullptr, &event);
                calc_alpha_time += getComputeTime(event);
            }

            queue.enqueueNDRangeKernel(update_guess_kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local), nullptr, &event);
            update_guess_time += getComputeTime(event);

            if (computeOneThreadedKernelsOnCPU)
            {
                queue.enqueueReadBuffer(rBuf, CL_TRUE, 0, r.size() * sizeof(double), r.data(), nullptr, &event);
                read_buffers_time += getComputeTime(event);

                const auto start = std::chrono::steady_clock::now();
                    new_r_dot_r = 0.0;
                    for (int i = 0; i < dimension; i++)
                    {
                        new_r_dot_r += r[i] * r[i];
                    }
                    r_length = sqrt(new_r_dot_r);
                 const auto end = std::chrono::steady_clock::now();

                 update_r_length_new_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

                queue.enqueueWriteBuffer(new_r_dot_r_Buf, CL_TRUE, 0, sizeof(double), &new_r_dot_r, nullptr, &event);
                read_buffers_time += getComputeTime(event);
            }
            else
            {
                queue.enqueueNDRangeKernel(update_r_length_new_kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1), nullptr, &event);
                update_r_length_new_time += getComputeTime(event);

                queue.enqueueReadBuffer(r_length_Buf, CL_TRUE, 0, sizeof(double), &r_length, nullptr, &event);
                read_buffers_time += getComputeTime(event);
            }

            queue.enqueueNDRangeKernel(update_direction_kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local), nullptr, &event);
            update_direction_time += getComputeTime(event);

            queue.enqueueNDRangeKernel(sync_r_dot_r_kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1), nullptr, &event);
            sync_r_dot_r_time += getComputeTime(event);

            iterations++;
        }

        queue.enqueueReadBuffer(xBuf, CL_TRUE, 0, x.size() * sizeof(double), x.data(), nullptr, &event);
        read_buffers_time += getComputeTime(event);

        total_kernel_time = init_time + update_r_length_old_time + update_A_times_p_time + calc_alpha_time + update_guess_time + update_r_length_new_time + update_direction_time + sync_r_dot_r_time + read_buffers_time;
    };


    auto timeInfo = std::make_unique<ScaledTimeInfo>();
    timeInfo->total_compute_time       = Timer::computeTime(computeLinearSystem);
    timeInfo->total_kernel_time        = Timer::toAppropriateMeasure(total_kernel_time);
    timeInfo->init_time                = Timer::toAppropriateMeasure(init_time);
    timeInfo->update_r_length_old_time = Timer::toAppropriateMeasure(update_r_length_old_time);
    timeInfo->update_A_times_p_time    = Timer::toAppropriateMeasure(update_A_times_p_time);
    timeInfo->calc_alpha_time          = Timer::toAppropriateMeasure(calc_alpha_time);
    timeInfo->update_guess_time        = Timer::toAppropriateMeasure(update_guess_time);
    timeInfo->update_r_length_new_time = Timer::toAppropriateMeasure(update_r_length_new_time);
    timeInfo->update_direction_time    = Timer::toAppropriateMeasure(update_direction_time);
    timeInfo->sync_r_dot_r_time        = Timer::toAppropriateMeasure(sync_r_dot_r_time);
    timeInfo->read_buffers_time        = Timer::toAppropriateMeasure(read_buffers_time);

    return { x, iterations, r_length, std::move(timeInfo) };
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

    auto timeInfo = std::make_unique<TimeInfo>();
    timeInfo->total_compute_time = Timer::computeTime(computeLinearSystem);

    return { x, static_cast<int>(result[0]), result[1], std::move(timeInfo) };
}

Result conjugateGradientCpu(const SparseMatrix& aSparseMatrix)
{
    const int dim = aSparseMatrix.getDimension();
    const int num_vals = aSparseMatrix.getValuesNum();
    const int* rows = aSparseMatrix.getRowIds();
    const int* cols = aSparseMatrix.getColIds();
    const double* A = aSparseMatrix.getValues();
    const double* b = aSparseMatrix.getVectorB();

    std::vector<double> x(aSparseMatrix.getDimension());
    int iterations;
    double residualLength;

    long long init_time = 0;
    long long update_r_length_old_time = 0;
    long long update_A_times_p_time = 0;
    long long calc_alpha_time = 0;
    long long update_guess_time = 0;
    long long update_r_length_new_time = 0;
    long long update_direction_time = 0;
    long long sync_r_dot_r_time = 0;
    //long long read_buffers_time = 0;

    auto computeLinearSystem = [&]()
    {
        auto start = std::chrono::steady_clock::now();

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

        auto end = std::chrono::steady_clock::now();

        init_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::steady_clock::now();
            old_r_dot_r = 0.0;
            for (int i = 0; i < dim; i++)
            {
                old_r_dot_r += r[i] * r[i];
            }
            r_length = sqrt(old_r_dot_r);

        end = std::chrono::steady_clock::now();

        update_r_length_old_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        int iteration = 0;
        while (iteration < 50000 && r_length >= 0.01)
        {
            start = std::chrono::steady_clock::now();
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
            end = std::chrono::steady_clock::now();

            update_A_times_p_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            start = std::chrono::steady_clock::now();
                Ap_dot_p = 0.0;
                for (int i = 0; i < dim; i++)
                {
                    Ap_dot_p += A_times_p[i] * p[i];
                }
                alpha = old_r_dot_r / Ap_dot_p;
            end = std::chrono::steady_clock::now();

            calc_alpha_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            start = std::chrono::steady_clock::now();
                for (int i = 0; i < dim; i++)
                {
                    x[i] += alpha * p[i];
                    r[i] -= alpha * A_times_p[i];
                }
            end = std::chrono::steady_clock::now();

            update_guess_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            start = std::chrono::steady_clock::now();
                new_r_dot_r = 0.0;
                for (int i = 0; i < dim; i++)
                {
                    new_r_dot_r += r[i] * r[i];
                }
                r_length = sqrt(new_r_dot_r);
            end = std::chrono::steady_clock::now();

            update_r_length_new_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            start = std::chrono::steady_clock::now();
                for (int i = 0; i < dim; i++)
                {
                    p[i] = r[i] + (new_r_dot_r / old_r_dot_r) * p[i];
                }
            end = std::chrono::steady_clock::now();

            update_direction_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            start = std::chrono::steady_clock::now();
                old_r_dot_r = new_r_dot_r;
            end = std::chrono::steady_clock::now();
            sync_r_dot_r_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            iteration++;
        }
        iterations = iteration;
        residualLength = r_length;
    };

    auto timeInfo = std::make_unique<ScaledTimeInfo>();
    timeInfo->total_compute_time = Timer::computeTime(computeLinearSystem);
    //timeInfo->total_kernel_time = Timer::toAppropriateMeasure(total_kernel_time);
    timeInfo->init_time = Timer::toAppropriateMeasure(init_time);
    timeInfo->update_r_length_old_time = Timer::toAppropriateMeasure(update_r_length_old_time);
    timeInfo->update_A_times_p_time = Timer::toAppropriateMeasure(update_A_times_p_time);
    timeInfo->calc_alpha_time = Timer::toAppropriateMeasure(calc_alpha_time);
    timeInfo->update_guess_time = Timer::toAppropriateMeasure(update_guess_time);
    timeInfo->update_r_length_new_time = Timer::toAppropriateMeasure(update_r_length_new_time);
    timeInfo->update_direction_time = Timer::toAppropriateMeasure(update_direction_time);
    timeInfo->sync_r_dot_r_time = Timer::toAppropriateMeasure(sync_r_dot_r_time);
    //timeInfo->read_buffers_time = Timer::toAppropriateMeasure(read_buffers_time);

    return { x, iterations, residualLength, std::move(timeInfo) };
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

    auto timeInfo = std::make_unique<TimeInfo>();
    timeInfo->total_compute_time = Timer::computeTime(computeLinearSystem);

    return { x, iterations, residualLength, std::move(timeInfo) };
}

std::vector<double> matrixVectorMultiplication(const SparseMatrix& aSparseMatrix, const std::vector<double>& aVector)
{
    assert(aSparseMatrix.getDimension() == aVector.size());
    std::vector<double> result;
    result.reserve(aVector.size());

    const int* rows = aSparseMatrix.getRowIds();
    const int* cols = aSparseMatrix.getColIds();
    const double* vals = aSparseMatrix.getValues();

    int activeRowId = 0;
    double value = 0.0;
    for (size_t i = 0; i < aSparseMatrix.getValuesNum();)
    {
        if (rows[i] == activeRowId)
        {
            value += vals[i] * aVector[cols[i]];
            i++;
        }
        else
        {
            activeRowId = rows[i];
            result.push_back(value);
            value = 0.0;
        }
    }

    return result;
}

}
