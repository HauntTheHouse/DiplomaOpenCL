#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <CL/opencl.hpp>

#include "SparseMatrix.h"
#include "Time.h"
#include "Utils.h"

void conjugateGradientCpu(int dim, int num_vals, double *x, int *rows, int *cols, double *A,
                          double *b, double *result)
{
    std::vector<double> r(dim);
    std::vector<double> A_times_p(dim);
    std::vector<double> p(dim);
    double alpha, r_length, old_r_dot_r, new_r_dot_r;
    int iteration;

//    int id = get_local_id(0);
//    int start_index = -1;
//    int end_index = -1;
    double Ap_dot_p;

    //printf("ID: %d\n", id);

    // for (int i = id; i < num_vals; i++)
    // {
    //     if ((rows[i] == id) && (start_index == -1))
    //     {
    //         start_index = i;
    //     }
    //     else if ((rows[i] == id + 1) && (end_index == -1))
    //     {
    //         end_index = i - 1;
    //         break;
    //     }
    //     else if ((i == num_vals - 1) && (end_index == -1))
    //     {
    //         end_index = i;
    //     }
    // }

//    start_index = 0;
//    end_index = num_vals - 1;

//    printf("ThreadID: %d, StartID: %d, EndID: %d\n", id, start_index, end_index);

    for (int i = 0; i < dim; ++i)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }
    // x[id] = 0.0;
    // r[id] = b[id];
    // p[id] = b[id];
//    barrier(CLK_LOCAL_MEM_FENCE);

//    if (id == 0)
    {
        old_r_dot_r = 0.0;
        for (int i = 0; i < dim; i++)
        {
            old_r_dot_r += r[i] * r[i];
        }
        r_length = sqrt(old_r_dot_r);
    }
//    barrier(CLK_LOCAL_MEM_FENCE);

    iteration = 0;
    while ((iteration < 10000) && (r_length >= 0.01))
    {
//        printf("Iteration: %d\n", iteration);
        int etalon = 0;
        int j = 0;

        for (int i = 0; i < dim; ++i)
        {
            // printf("etalon = %d\n", etalon);
            A_times_p[i] = 0.0;
            // int j = i;
            while (etalon == rows[j])
            {
                // printf("\trows[%d] = %d\n", j, rows[j]);
                A_times_p[i] += A[j] * p[cols[j]];
                j++;
            }
            etalon++;
        }
        // A_times_p[id] = 0.0;
        // for (int i = start_index; i <= end_index; i++)
        // {
        //     A_times_p[id] += A[i] * p[cols[i]];
        // }
//        barrier(CLK_LOCAL_MEM_FENCE);

//        if (id == 0)
        {
            Ap_dot_p = 0.0;
            for (int i = 0; i < dim; i++)
            {
                Ap_dot_p += A_times_p[i] * p[i];
            }
            alpha = old_r_dot_r/Ap_dot_p;
        }
//        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < dim; i++)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * A_times_p[i];
        }
        // x[id] += alpha * p[id];
        // r[id] -= alpha * A_times_p[id];
//        barrier(CLK_LOCAL_MEM_FENCE);

//        if (id == 0)
        {
            new_r_dot_r = 0.0;
            for (int i = 0; i < dim; i++)
            {
                new_r_dot_r += r[i] * r[i];
            }
            r_length = sqrt(new_r_dot_r);
        }
//        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < dim; i++)
        {
            p[i] = r[i] + (new_r_dot_r/old_r_dot_r) * p[i];
        }
        // p[id] = r[id] + (new_r_dot_r/old_r_dot_r) * p[id];
//        barrier(CLK_LOCAL_MEM_FENCE);

        old_r_dot_r = new_r_dot_r;

//        if (id == 0)
        {
            iteration++;
        }
//        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[0] = (double)iteration;
    result[1] = r_length;
}

int main()
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

    SparseMatrix sparseMatrix("data/sparse_matrix_112.txt");
    //SparseMatrix sparseMatrix("data/bcsstm12.mtx");
    sparseMatrix.fillVectorBFullyWithConcreteValue(200.0);

    std::ofstream stream;
    stream.open("out.txt");
    sparseMatrix.print(stream);
    stream.close();

    std::string method = "conjugateGradient";
//    std::string method = "steepestDescent";

    std::cout << "Algorithm that solves linear equation: " << method << std::endl;
    std::string src = Utils::readFileToString("kernels/" + method + ".cl");

    cl::Program::Sources sources;
    sources.push_back(src);

    context = cl::Context(device);
    program = cl::Program(context, sources);

    auto err = program.build();
//    std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
    if (err != CL_BUILD_SUCCESS)
    {
        std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
               << "\nBuild Log:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }

    const auto numValues = sparseMatrix.getValuesNum();
    const auto dimension = sparseMatrix.getDimension();

    std::vector<double> x(dimension);
    std::vector<double> result(2);

    cl::Buffer rowsBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(int), sparseMatrix.getRowIds());
    cl::Buffer colsBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(int), sparseMatrix.getColIds());
    cl::Buffer valuesBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, numValues * sizeof(double), sparseMatrix.getValues());
    cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, dimension * sizeof(double), sparseMatrix.getVectorB());
    cl::Buffer xBuf(context, CL_MEM_READ_WRITE, dimension * sizeof(double));
    cl::Buffer resultBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 2 * sizeof(double));

    cl::Kernel kernel(program, method.c_str());

    int kernelWorkGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    int deviceMaxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    std::cout << "KERNEL_WORK_GROUP_SIZE: " << kernelWorkGroupSize << std::endl;
    std::cout << "DEVICE_MAX_WORK_GROUP_SIZE: " << deviceMaxWorkGroupSize << std::endl;

    if (dimension > kernelWorkGroupSize)
        return -1;

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
//        kernel.setArg(5, rowsBuf);
//        kernel.setArg(6, colsBuf);
//        kernel.setArg(7, valuesBuf);
//        kernel.setArg(8, bBuf);
//        kernel.setArg(9, resultBuf);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(dimension), cl::NDRange(dimension));

        queue.enqueueReadBuffer(xBuf, CL_TRUE, 0, x.size() * sizeof(double), x.data());
        queue.enqueueReadBuffer(resultBuf, CL_TRUE, 0, result.size() * sizeof(double), result.data());
    };

    const auto func = [&]()
    {
        conjugateGradientCpu(dimension, numValues, x.data(),
                                    sparseMatrix.getRowIds(),
                                    sparseMatrix.getColIds(),
                                    sparseMatrix.getValues(),
                                    sparseMatrix.getVectorB(),
                                    result.data());
    };

    const auto measuredTime = Time::compute(computeLinearSystem);
    //const auto measuredTime = Time::compute(func);

    for (const auto val : x)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Iterations: " << static_cast<int>(result[0]) << std::endl;
    std::cout << "Residual length: " << result[1] << std::endl << std::endl;
    std::cout << "Compute time: " << measuredTime.value << " " << Time::toString(measuredTime.measure) << std::endl;
}
