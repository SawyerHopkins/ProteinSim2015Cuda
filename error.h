#ifndef ERROR_H
#define ERROR_H
#include <iostream>
#include <exception>

namespace debugging
{

	class error : public std::exception
	{

		public:

			/**
			 * @brief Throw when system initial conditions cannot be resolved.
			 */
			static void throwInitializationError();
			/**
			 * @brief Throw when trying to access a non existant cell.
			 * @param cx,cy,cz The cell to be accessed.
			 */
			static void throwCellBoundsError(int cx, int cy, int cz);
			/**
			 * @brief Throw when the particle is outside the box after PBC check.
			 * @param x,y,z The position of the particle.
			 */
			static void throwParticleBoundsError(int x, int y, int z);
			/**
			 * @brief Throw when particles are closer than the accepted tolerance.
			 * @param nameI Name of the index particle.
			 * @param nameJ Name of the force particle.
			 * @param r The distance between the particles.
			 */
			static void throwParticleOverlapError(int nameI, int nameJ, double r);
			/**
			 * @brief Throw when force is given value NaN.
			 */
			static void throwInfiniteForce();

	};

}

#endif // ERROR_H
