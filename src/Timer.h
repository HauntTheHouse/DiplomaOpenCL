#pragma once

#include <chrono>
#include <unordered_map>
#include <string>
#include <iostream>

namespace Timer
{
    enum class Measure
    {
        NANOSECONDS = 0,
        MICROSECONDS = 1,
        MILLISECONDS = 2,
        SECONDS = 3,
    };

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
    struct ComputedTime
    {
        long double value{ 0 };
        Measure measure{ Measure::NANOSECONDS };

        friend std::ostream& operator<<(std::ostream& os, const ComputedTime& computedTime)
        {
            os << computedTime.value << '\t' << Timer::toString(computedTime.measure);
            return os;
        }
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

    inline long double toNanoseconds(const ComputedTime& time)
    {
        const auto measureVal = static_cast<size_t>(time.measure);
        auto count = time.value;

        for (size_t i = measureVal; i > 0; --i)
        {
            count *= 1000.0;
        }
        return count;
    }

    inline long double calcPercentage(const ComputedTime& value, const ComputedTime& total)
    {
        return toNanoseconds(value) / toNanoseconds(total) * 100.0;
    }

};
