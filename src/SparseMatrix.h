#pragma once

#include <string>
#include <fstream>
#include <vector>

class SparseMatrix
{
public:
    SparseMatrix() = default;
    SparseMatrix(const std::string& pathToMatrix);
    ~SparseMatrix();

    void open(const std::string& pathToMatrix);
    void clear();

    void print(std::ostream& stream);

    int getDimension() const { return m_Dimension; }
    int getValuesNum() const { return m_NumValues; }

    const int* getRowIds() const { return m_RowIds.data(); }
    const int* getColIds() const { return m_ColIds.data(); }
    const double* getValues() const { return m_Values.data(); }
    const double* getVectorB() const { return m_B.data(); }

    void fillVectorBWithRandomValues(double minValue, double maxValue);
    void fillVectorBWithValue(double value);

private:
    void sort();

    inline static const std::string MATRIX_MARKET{"%%MatrixMarket"};
    inline static const std::string MATRIX{"matrix"};
    inline static const std::string COORDINATE{"coordinate"};
    inline static const std::string REAL{"real"};
    inline static const std::string COMPLEX{"complex"};
    inline static const std::string SYMMETRIC{"symmetric"};
    inline static const std::string GENERAL{"general"};

    std::ifstream m_MatrixFile;

    bool m_IsReal{ true };
    bool m_IsSymmetric{ false };

    int m_Dimension{ 0 };
    int m_NumValues{ 0 };

    std::vector<int> m_RowIds;
    std::vector<int> m_ColIds;
    std::vector<double> m_Values;
    std::vector<double> m_B;
};
