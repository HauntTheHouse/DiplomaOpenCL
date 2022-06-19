#pragma once

#include <optional>
#include <iostream>

#include "Timer.h"

struct TimeInfo
{
	Timer::ComputedTime total_compute_time;

	virtual void print(std::ostream& out) const
	{
		out << "Total compute time: " << total_compute_time << '\n';
	}

	friend std::ostream& operator<<(std::ostream& out, const TimeInfo& timeInfo)
	{
		timeInfo.print(out);
		return out;
	}
};

struct NonScaledTimeInfo : public TimeInfo
{
	Timer::ComputedTime total_kernel_time;
	Timer::ComputedTime kernel_compute_time;
	Timer::ComputedTime read_buffer_time;

	virtual void print(std::ostream& out) const
	{
		out << "Total compute time:  " << total_compute_time << '\n';
		out << "Total kernel time:   " << total_kernel_time << '\n';
		out << '\n';
		out << "Kernel compute time: " << kernel_compute_time << '\t' << Timer::calcPercentage(kernel_compute_time, total_kernel_time) << "%\n";
		out << "Read buffer time:    " << read_buffer_time << '\t' << Timer::calcPercentage(read_buffer_time, total_kernel_time) << "%\n";
	}
};

struct ScaledTimeInfo : public TimeInfo
{
	std::optional<Timer::ComputedTime> total_kernel_time;
	std::optional<Timer::ComputedTime> read_buffers_time;
	Timer::ComputedTime init_time;
	Timer::ComputedTime update_r_length_old_time;
	Timer::ComputedTime update_A_times_p_time;
	Timer::ComputedTime calc_alpha_time;
	Timer::ComputedTime update_guess_time;
	Timer::ComputedTime update_r_length_new_time;
	Timer::ComputedTime update_direction_time;
	Timer::ComputedTime sync_r_dot_r_time;

	virtual void print(std::ostream& out) const
	{
		Timer::ComputedTime total_time;

		out << "Total compute time:          " << total_compute_time << '\n';
		if (total_kernel_time.has_value())
		{
			out << "Total kernel time:           " << total_kernel_time.value() << '\n';
			total_time = total_kernel_time.value();
		}
		else
		{
			total_time = total_compute_time;
		}
		out << '\n';
		out << "Init time:                   " << init_time << '\t' << Timer::calcPercentage(init_time, total_time) << "%\n";
		out << "Init residual length time:   " << update_r_length_old_time << '\t' << Timer::calcPercentage(update_r_length_old_time, total_time) << "%\n";
		out << "Calculate A*p time:          " << update_A_times_p_time << '\t' << Timer::calcPercentage(update_A_times_p_time, total_time) << "%\n";
		out << "Calculate alpha time:        " << calc_alpha_time << '\t' << Timer::calcPercentage(calc_alpha_time, total_time) << "%\n";
		out << "Update guess time:           " << update_guess_time << '\t' << Timer::calcPercentage(update_guess_time, total_time) << "%\n";
		out << "Update residual length time: " << update_r_length_new_time << '\t' << Timer::calcPercentage(update_r_length_new_time, total_time) << "%\n";
		out << "Sync r*r time:               " << sync_r_dot_r_time << '\t' << Timer::calcPercentage(sync_r_dot_r_time, total_time) << "%\n";
		if (read_buffers_time.has_value())
			out << "Read buffers time:           " << read_buffers_time.value() << '\t' << Timer::calcPercentage(read_buffers_time.value(), total_time) << "%\n";
	}
};
