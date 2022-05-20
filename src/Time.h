#pragma once

#include <chrono>
#include <unordered_map>
#include <string>

namespace Time
{
    enum class Measure
    {
        NANOSECONDS,
        MICROSECONDS,
        MILISECONDS,
        SECONDS,
    };

    struct Result
    {
        int64_t value;
        Measure measure;
    };

    template<typename T>
    Result compute(const T& funcToCompute)
    {
        const auto start = std::chrono::steady_clock::now();
        funcToCompute();
        const auto end = std::chrono::steady_clock::now();

        Measure measure = Measure::NANOSECONDS;
        auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        const auto measuresNum = static_cast<size_t>(Measure::SECONDS) + 1;
        for (size_t i = 0; i < measuresNum; ++i)
        {
            if (count < 1000)
                break;
            count /= 1000;
            measure = static_cast<Measure>(static_cast<size_t>(measure) + 1);
        }

        return { count, measure };
    }

    std::string toString(Measure measure)
    {
        switch (measure)
        {
        case Time::Measure::NANOSECONDS:
            return "nanoseconds";
        case Time::Measure::MICROSECONDS:
            return "microseconds";
        case Time::Measure::MILISECONDS:
            return "miliseconds";
        case Time::Measure::SECONDS:
            return "seconds";
        default:
            return "";
        }
    }
};
