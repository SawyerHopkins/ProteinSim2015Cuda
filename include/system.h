#ifndef SYSTEM_H
#define SYSTEM_H
#include "integrator.h"

namespace simulation
{

	/**
	 * @class system
	 * @author Sawyer Hopkins
	 * @date 06/27/15
	 * @file system.h
	 * @brief Frontend wrapper for running simulations.
	 */
	class system
	{

		private:

			/********************************************//**
			*-----------------SYSTEM VARIABLES---------------
			 ***********************************************/

			//Trial name
			std::string trialName;

			//Information about the system.
			int nParticles;
			double concentration;
			int boxSize;
			int cellSize;
			int cellScale;
			double temp;
			double currentTime;
			double dTime;

			//Random number seed.
			int seed;

			//System entities.
			particle** particles;
			cell**** cells;

			//System integrator.
			integrators::I_integrator* integrator;
			physics::forces* sysForces;

			/********************************************//**
			*-------------------SYSTEM INIT------------------
			 ***********************************************/

			/**
			 * @brief Creates the cell system.
			 * @param numCells The number of cells to be created.
			 * @param scale The number of cells in each dimension. (numCells^1/3)
			 */
			void initCells(int numCells, int scale);
			/**
			 * @brief Creates an initial uniform distribution of particles.
			 * @param r The radius of the particles
			 * @param m The mass of the particles.
			 */
			void initParticles(double r, double m);
			/**
			 * @brief Creates a maxwell distribution of velocities for the system temperature.
			 * @param gen The random generator the initalize particles.
			 * @param distribution The distribution for the particles.
			 */
			void maxwellVelocityInit(std::mt19937* gen, std::uniform_real_distribution<double>* distribution);
			/**
			 * @brief Fixes any particle overlap in the random initalization.
			 * @param gen The random generator the initalize particles.
			 * @param distribution The distribution for the particles.
			 */
			void initCheck(std::mt19937* gen, std::uniform_real_distribution<double>* distribution);

			/**
			 * @brief Get input for working directory. Create if needed.
			 * @return The working directory 
			 */
			std::string runSetup();

			/********************************************//**
			*-----------------SYSTEM HANDLING----------------
			 ***********************************************/

			/**
			 * @brief Updates the cells that the particles are located in.
			 * @return 
			 */
			void updateCells();

		public:

			//Header Version.
			const int version = 1;

			/********************************************//**
			*---------------SYSTEM CONSTRUCTION--------------
			 ***********************************************/

			/**
			 * @brief Constructs the particle system.
			 * @return Nothing.
			 */
			system(configReader::config* cfg, integrators::I_integrator* sysInt, physics::forces* sysFcs);
			/**
			 * @brief Destructs the particle system.
			 * @return Nothing.
			 */
			~system();

			/********************************************//**
			*-----------------SYSTEM GETTERS-----------------
			 ***********************************************/

			/**
			 * @brief Gets the number of particles in the system.
			 * @return Number of particles.
			 */
			const int getNParticles() const { return nParticles; }
			/**
			 * @brief Gets the length of the system box.
			 * @return length of the system box.
			 */
			const int getBoxSize() const { return boxSize; }
			/**
			 * @brief Gets the length of a system cell.
			 * @return cellSize.
			 */
			const int getCellSize() const { return cellSize; }

			/********************************************//**
			*-----------------SYSTEM HANDLING----------------
			 ***********************************************/

			/**
			 * @brief Runs the system.
			 * @param endTime When to stop running the simulation.
			 */
			void run(double endTime);

			/********************************************//**
			*------------------SYSTEM OUTPUT-----------------
			 ***********************************************/

			/**
			 * @brief Writes the temperature of the system.
			 */
			void writeInitTemp();
			/**
			 * @brief Writes the position of a particle.
			 * @param index The index of the particle to write.
			 */
			void writePosition(int index) { particles[index]->writePosition(); }
			/**
			 * @brief Writes the system to file.
			 * @param name The name of the file to write to.
			 */
			void writeSystem(std::string name);
			/**
			 * @brief Write the initial system parameters.
			 */
			void writeSystemInit();

			/********************************************//**
			*-----------------SYSTEM RECOVERY----------------
			 ***********************************************/

			/**
			 * @brief Recover a system state from output files.
			 * @param settings The location of the settings file.
			 * @param sysState The location of the system file.
			 */
			void loadFromFile(std::string settings, std::string sysState);

			/********************************************//**
			*-----------------SYSTEM ANALYSIS----------------
			 ***********************************************/

			/**
			 * @brief Get the number of clusters in the system.
			 * @return Return the number of clusters
			 */
			int numClusters();

	};

}

#endif // SYSTEM_H
