#include "Timer.h"

#include <chrono>

template<typename T, typename ...Args>
Timer::ComputedTime Timer::computeTime(T&& funcToCompute, Args&& ...arguments)
{
    const auto time = toAppropriateMeasure(computeTimeInNanoseconds(std::forward<T>(funcToCompute), std::forward<Args>(arguments)...));
    return time;
}

template<typename T, typename ...Args>
long long Timer::computeTimeInNanoseconds(T&& funcToCompute, Args&& ...arguments)
{
    const auto start = std::chrono::steady_clock::now();
    funcToCompute(std::forward<Args>(arguments)...);
    const auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

template<typename T>
Timer::ComputedTime Timer::toAppropriateMeasure(T nanoseconds)
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