#include "Timer.h"

#include <iostream>

std::ostream& Timer::operator<<(std::ostream& os, const ComputedTime& computedTime)
{
    os << computedTime.value << '\t' << Timer::toString(computedTime.measure);
    return os;
}

long double Timer::calcPercentage(const ComputedTime& value, const ComputedTime& total)
{
    return toNanoseconds(value) / toNanoseconds(total) * 100.0;
}

std::string Timer::toString(Measure measure)
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

long double Timer::toNanoseconds(const ComputedTime& time)
{
    const auto measureVal = static_cast<size_t>(time.measure);
    auto count = time.value;

    for (size_t i = measureVal; i > 0; --i)
    {
        count *= 1000.0;
    }
    return count;
}
