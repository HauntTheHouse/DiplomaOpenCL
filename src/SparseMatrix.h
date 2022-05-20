#pragma once

#include <string>
#include <fstream>
#include <vector>

class SparseMatrix
{
public:
    explicit SparseMatrix(const std::string& pathToMatrix);
    ~SparseMatrix();

    void open(const std::string& pathToMatrix);
    void clear();

    void print(std::ostream& stream);

    int getDimension() const { return m_Dimension; }
    int getValuesNum() const { return m_NumValues; }
    int* getRowIds() { return m_RowIds.data(); }
    int* getColIds() { return m_ColIds.data(); }
    double* getValues() { return m_Values.data(); }
    double* getVectorB() { return m_B.data(); }

    void fillVectorBWithRandomValues(double minValue, double maxValue);
    void fillVectorBFullyWithConcreteValue(double value);

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
