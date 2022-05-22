#include "SparseMatrix.h"

#include <sstream>
#include <ctime>
#include <cassert>
#include <iostream>

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

    m_MatrixFile.open(pathToMatrix);
    if (!m_MatrixFile.is_open())
        throw std::invalid_argument("This file doesn't exist");

    std::string header;
    std::getline(m_MatrixFile, header);
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
            m_IsReal = true;
        else if (str == COMPLEX)
            m_IsReal = false;
        else
            throw std::invalid_argument("Matrix should has real or complex numbers");

        headerStream >> str;
        if (str == SYMMETRIC)
            m_IsSymmetric = true;
        else if (str == GENERAL)
            m_IsReal = false;
        else
            throw std::invalid_argument("Matrix should be symmetric or general");

        std::getline(m_MatrixFile, header);
        headerStream = std::stringstream(header);
    }

    int rowNum, colNum;
    headerStream >> rowNum >> colNum >> m_NumValues;
    assert(rowNum == colNum);
    m_Dimension = rowNum;

    if (m_IsSymmetric)
        m_NumValues = m_NumValues * 2 - m_Dimension;

    m_RowIds.reserve(m_NumValues);
    m_ColIds.reserve(m_NumValues);
    m_Values.reserve(m_NumValues);
    m_B.reserve(m_Dimension);

    bool toSort = false;
    while (!m_MatrixFile.eof())
    {
        int rowId, colId;
        double value;
        m_MatrixFile >> rowId >> colId >> value;

//        if (!toSort && !m_RowIds.empty() && rowId < m_RowIds.back())
//            toSort = true;

        m_RowIds.push_back(rowId - 1);
        m_ColIds.push_back(colId - 1);
        m_Values.push_back(value);

        if (m_IsSymmetric && rowId != colId)
        {
            m_RowIds.push_back(colId - 1);
            m_ColIds.push_back(rowId - 1);
            m_Values.push_back(value);
        }
    }

    if (m_IsSymmetric)
        sort();

    m_MatrixFile.close();
}

void SparseMatrix::clear()
{
    m_RowIds.clear();
    m_ColIds.clear();
    m_Values.clear();
    m_B.clear();
}

void SparseMatrix::fillVectorBWithRandomValues(double minValue, double maxValue)
{
    srand(time(nullptr));
    for (int i = 0; i < m_Dimension; ++i)
    {
        m_B.push_back(rand() / static_cast<double>(RAND_MAX) * (maxValue - minValue) + minValue);
    }
}

void SparseMatrix::fillVectorBWithValue(double value)
{
    for (int i = 0; i < m_Dimension; ++i)
    {
        m_B.push_back(value);
    }
}

void SparseMatrix::sort()
{
    int int_swap, index = 0;
    double double_swap;
    for(int i = 0; i < m_NumValues; i++)
    {
        for(int j = index; j < m_NumValues; j++)
        {
            if(m_RowIds[j] == i)
            {
                if(j == index)
                {
                    index++;
                }
                else if(j > index)
                {
                    int_swap = m_RowIds[index];
                    m_RowIds[index] = m_RowIds[j];
                    m_RowIds[j] = int_swap;
                    int_swap = m_ColIds[index];
                    m_ColIds[index] = m_ColIds[j];
                    m_ColIds[j] = int_swap;
                    double_swap = m_Values[index];
                    m_Values[index] = m_Values[j];
                    m_Values[j] = double_swap;
                    index++;
                }
            }
        }
    }
}

void SparseMatrix::print(std::ostream& stream)
{
    stream.precision(14);
    stream << m_Dimension << ' ' << m_Dimension << ' ' << m_NumValues << '\n';
    for (int i = 0; i < m_NumValues; ++i)
    {
        stream << m_RowIds[i] << ' ' << m_ColIds[i] << ' ' << m_Values[i] << '\n';
    }
}
