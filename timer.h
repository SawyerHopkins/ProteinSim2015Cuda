#ifndef TIMER_H
#define TIMER_H
#include <chrono>
#include <math.h>

namespace debugging
{

	class timer
	{

		private:

			std::chrono::system_clock::time_point start_time;
			std::chrono::system_clock::time_point stop_time;

		public:

			timer();
			~timer();

			/**
			 * @brief Start the timer.
			 */
			void start() { start_time = std::chrono::system_clock::now(); }
			/**
			 * @brief End the timer.
			 */
			void stop() { stop_time = std::chrono::system_clock::now(); }

			double getElapsedSeconds() 
			{
				std::chrono::system_clock::duration diff = stop_time-start_time;
				return (diff.count() / pow(10,9));
			}

	};

}

#endif // TIMER_H
