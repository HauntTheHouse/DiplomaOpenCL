#pragma once

#include <string>
#include <vector>

class SparseMatrix
{
public:
    SparseMatrix() = default;
    SparseMatrix(const std::string& aPathToMatrix);
    ~SparseMatrix();

    void open(const std::string& aPathToMatrix);
    void clear();

    void print(std::ostream& aStream);

    int getDimension() const { return mDimension; }
    int getValuesNum() const { return mNumValues; }

    const int* getRowIds() const { return mRowIds.data(); }
    const int* getColIds() const { return mColIds.data(); }
    const double* getValues() const { return mValues.data(); }
    const double* getVectorB() const { return mB.data(); }

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

    bool mIsReal{ true };
    bool mIsSymmetric{ false };

    int mDimension{ 0 };
    int mNumValues{ 0 };

    std::vector<int> mRowIds;
    std::vector<int> mColIds;
    std::vector<double> mValues;
    std::vector<double> mB;
};
