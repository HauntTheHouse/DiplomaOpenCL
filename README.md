# DiplomaOpenCL
This is the project created during the writing of my diploma work of [NTUU KPI](https://en.wikipedia.org/wiki/Igor_Sikorsky_Kyiv_Polytechnic_Institute) of IASA faculty. The topic of this paper is "SLAE solving procedures with sparse matrices using both GPU and CPU compution".

It is a program that allows you to solve a system of linear equations with sparse matrices using GPU and CPU calculations.

To calculate systems of linear equations, this program uses the _conjugate gradient method_ and _steepest descent method_ (not recommended to use because the method often diverges). The basis of these algorithms was taken from the book "OpenCL in Action: How to Accelerate Graphics and Computations". I also added my modifications to these algorithms.

This project was written using C++17 and framework OpenCL. Also it uses ViennaCL library for comparison with my algorithms.

## Requeriments
To build this project, you need a compiler that supports C++17, CMake and [OpenCL binaries](https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/choose-download.html).
> Although the program should be cross-platform and support different operation systems, it was tested only on Windows 10.

## Usage
To build the project you need to do following steps:
```
git clone --recursive https://github.com/HauntTheHouse/DiplomaOpenCL.git
cd DiplomaOpenCL
cmake -S . -B build
cmake --build build --target DiplomaOpenCL --config Release
```
