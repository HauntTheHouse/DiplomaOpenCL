#pragma once

#include <string>

namespace Timer
{
    enum class Measure
    {
        NANOSECONDS = 0,
        MICROSECONDS = 1,
        MILLISECONDS = 2,
        SECONDS = 3,
    };

    struct ComputedTime
    {
        long double value{ 0 };
        Measure measure{ Measure::NANOSECONDS };

        friend std::ostream& operator<<(std::ostream& os, const ComputedTime& computedTime);
    };

    template<typename T, typename ...Args>
    ComputedTime computeTime(T&& funcToCompute, Args&&... arguments);
    template<typename T, typename ...Args>
    long long computeTimeInNanoseconds(T&& funcToCompute, Args&&... arguments);

    long double calcPercentage(const ComputedTime& value, const ComputedTime& total);

    std::string toString(Measure measure);
    template<typename T>
    ComputedTime toAppropriateMeasure(T nanoseconds);
    long double toNanoseconds(const ComputedTime& time);

};

#include "Timer.inl"
