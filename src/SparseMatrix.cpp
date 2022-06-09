#include "SparseMatrix.h"

#include <sstream>
#include <fstream>
#include <iostream>
#include <ctime>
#include <cassert>

SparseMatrix::SparseMatrix(const std::string& pathToMatrix)
{
    open(pathToMatrix);
}

SparseMatrix::~SparseMatrix()
{
    clear();
}

void SparseMatrix::open(const std::string& pathToMatrix)
{
    clear();

    std::ifstream matrixFile(pathToMatrix);
    if (!matrixFile.is_open())
        throw std::invalid_argument("This file doesn't exist");

    std::string header;
    std::getline(matrixFile, header);
    std::stringstream headerStream(header);

    if (header.front() == '%')
    {
        std::string str;
        headerStream >> str;
        if (str != MATRIX_MARKET)
            throw std::invalid_argument("No MatrixMarket in header");

        headerStream >> str;
        if (str != MATRIX)
            throw std::invalid_argument("File must contain the matrix");

        headerStream >> str;
        if (str != COORDINATE)
            throw std::invalid_argument("Matrix should be in coordinate format");

        headerStream >> str;
        if (str == REAL)
            mIsReal = true;
        else if (str == COMPLEX)
            mIsReal = false;
        else
            throw std::invalid_argument("Matrix should has real or complex numbers");

        headerStream >> str;
        if (str == SYMMETRIC)
            mIsSymmetric = true;
        else if (str == GENERAL)
            mIsReal = false;
        else
            throw std::invalid_argument("Matrix should be symmetric or general");

        std::getline(matrixFile, header);
        headerStream = std::stringstream(header);
    }

    int rowNum, colNum;
    headerStream >> rowNum >> colNum >> mNumValues;
    assert(rowNum == colNum);
    mDimension = rowNum;

    if (mIsSymmetric)
        mNumValues = mNumValues * 2 - mDimension;

    mRowIds.reserve(mNumValues);
    mColIds.reserve(mNumValues);
    mValues.reserve(mNumValues);
    mB.reserve(mDimension);

    bool toSort = false;
    while (!matrixFile.eof())
    {
        int rowId, colId;
        double value;
        matrixFile >> rowId >> colId >> value;

//        if (!toSort && !mRowIds.empty() && rowId < mRowIds.back())
//            toSort = true;

        mRowIds.push_back(rowId - 1);
        mColIds.push_back(colId - 1);
        mValues.push_back(value);

        if (mIsSymmetric && rowId != colId)
        {
            mRowIds.push_back(colId - 1);
            mColIds.push_back(rowId - 1);
            mValues.push_back(value);
        }
    }

    if (mIsSymmetric)
        sort();
}

void SparseMatrix::clear()
{
    mRowIds.clear();
    mColIds.clear();
    mValues.clear();
    mB.clear();
}

void SparseMatrix::fillVectorBWithRandomValues(double minValue, double maxValue)
{
    srand(time(nullptr));
    for (int i = 0; i < mDimension; ++i)
    {
        mB.push_back(rand() / static_cast<double>(RAND_MAX) * (maxValue - minValue) + minValue);
    }
}

void SparseMatrix::fillVectorBWithValue(double value)
{
    for (int i = 0; i < mDimension; ++i)
    {
        mB.push_back(value);
    }
}

void SparseMatrix::sort()
{
    int int_swap, index = 0;
    double double_swap;
    for(int i = 0; i < mNumValues; i++)
    {
        for(int j = index; j < mNumValues; j++)
        {
            if(mRowIds[j] == i)
            {
                if(j == index)
                {
                    index++;
                }
                else if(j > index)
                {
                    int_swap = mRowIds[index];
                    mRowIds[index] = mRowIds[j];
                    mRowIds[j] = int_swap;
                    int_swap = mColIds[index];
                    mColIds[index] = mColIds[j];
                    mColIds[j] = int_swap;
                    double_swap = mValues[index];
                    mValues[index] = mValues[j];
                    mValues[j] = double_swap;
                    index++;
                }
            }
        }
    }
}

void SparseMatrix::print(std::ostream& stream)
{
    stream.precision(14);
    stream << mDimension << ' ' << mDimension << ' ' << mNumValues << '\n';
    for (int i = 0; i < mNumValues; ++i)
    {
        stream << mRowIds[i] << ' ' << mColIds[i] << ' ' << mValues[i] << '\n';
    }
}
