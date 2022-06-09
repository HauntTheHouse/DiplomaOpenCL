#pragma once

#include <chrono>
#include <unordered_map>
#include <string>

namespace Timer
{
    enum class Measure
    {
        NANOSECONDS,
        MICROSECONDS,
        MILLISECONDS,
        SECONDS,
    };

    struct ComputedTime
    {
        long double value;
        Measure measure;
    };

    template<typename T, typename ...Args>
    ComputedTime computeTime(T&& funcToCompute, Args&&... arguments)
    {
        const auto start = std::chrono::steady_clock::now();
        funcToCompute(arguments...);
        const auto end = std::chrono::steady_clock::now();

        const auto time = convertNanosecondsToAppropriateMeasure(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        return time;
    }

    template<typename T>
    ComputedTime convertNanosecondsToAppropriateMeasure(T nanoseconds)
    {
        Measure measure = Measure::NANOSECONDS;
        auto count = static_cast<long double>(nanoseconds);

        const auto measuresNum = static_cast<size_t>(Measure::SECONDS) + 1;
        for (size_t i = 0; i < measuresNum; ++i)
        {
            if (count < 1000.0)
                break;
            count /= 1000.0;
            measure = static_cast<Measure>(static_cast<size_t>(measure) + 1);
        }
        return { count, measure };
    }

    inline std::string toString(Measure measure)
    {
        switch (measure)
        {
        case Measure::NANOSECONDS:
            return "nanoseconds";
        case Measure::MICROSECONDS:
            return "microseconds";
        case Measure::MILLISECONDS:
            return "milliseconds";
        case Measure::SECONDS:
            return "seconds";
        default:
            return "";
        }
    }
};
