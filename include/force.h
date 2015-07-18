#ifndef FORCE_H
#define FORCE_H
#include <ctime>
#include <omp.h>
#include "cell.h"
#include "config.h"

namespace physics
{

	/********************************************//**
	*-----------------FORCE INTERFACE----------------
	 ***********************************************/

	/**
	 * @class IForce
	 * @author Sawyer Hopkins
	 * @date 06/27/15
	 * @file force.h
	 * @brief A generic force container.
	 */
	class IForce
	{

		protected:

			std::string name;

		public:

			//Header Version.
			const int version = 1;

			/**
			 * @brief Virtual methods for forces of various parameters.
			 * @param index The index particle to find the force on.
			 * @param nPart The number of particles in the system.
			 * @param boxSize The size of the system.
			 * @param time The current system time.
			 * @param itemCell The cell containing the index particle. 
			 * @param items All particles in the system.
			 */
			virtual void getAcceleration(int index, int nPart, int boxSize, double time, simulation::cell* itemCell, simulation::particle** items)=0;

			/**
			 * @brief Flag for a force dependent time.
			 * @return True for time dependent. False otherwise. 
			 */
			virtual bool isTimeDependent()=0;

			/**
			 * @brief Get the name of the force for logging purposes.
			 * @return 
			 */
			std::string getName() { return name; }

	};

	/********************************************//**
	*----------------FORCE MANAGEMENT----------------
	 ***********************************************/

	/**
	 * @class forces
	 * @author Sawyer Hopkins
	 * @date 06/27/15
	 * @file force.h
	 * @brief Management system for a collection of forces.
	 */
	class forces
	{
		private:

			//A vector of all forces in the system.
			std::vector<IForce*> flist;
			//Flagged if flist contains a time dependant force.
			bool timeDependent;

		public:

			/**
			 * @brief Creates the force management system.
			 */
			forces();
			/**
			 * @brief Releases the management system.
			 */
			~forces();

			/**
			 * @brief Adds a force to the management system.
			 * @param f The force to add. Must implement IForce interface.
			 */
			void addForce(IForce* f);

			/**
			 * @brief Find the net force on all particles in the system.  
			 * @param nPart The number of particles in the system.
			 * @param boxSize The size of the system.
			 * @param time The current system time.
			 * @param cells The system cell manager.
			 * @param items The particles in the system.
			 */
			void getAcceleration(int nPart, int boxSize, double time, simulation::cell**** cells, simulation::particle** items);

			/**
			 * @brief Checks if the system contains a time dependent force.
			 * @return True if time dependent. False otherwise.
			 */
			bool isTimeDependent() { return timeDependent; }

			void setNumThreads(int num) { if (num > 0) {omp_set_num_threads(num);} }
			void setDynamic(int num) { omp_set_dynamic(num); }
			void setDevice(int num) { omp_set_default_device(num); }

			//Iterators

			/**
			 * @brief Gets the beginning iterator of the force list.
			 * @return flist.begin().
			 */
			std::vector<IForce*>::iterator getBegin() { return flist.begin(); }
			/**
			 * @brief Gets the end iterator of the force list.
			 * @return flist.end();
			 */
			std::vector<IForce*>::iterator getEnd() { return flist.end(); }

	};

	/********************************************//**
	*------------------AO POTENTIAL------------------
	 ***********************************************/

	/**
	 * @class AOPotential
	 * @author Sawyer Hopkins
	 * @date 06/27/15
	 * @file force.h
	 * @brief Drag force.
	 */
	class AOPotential : public IForce
	{

	private:

			//Variables vital to the force.
			double gamma;
			double cutOff;
			double dt;

			//Secondary variables.
			double coEff1;
			double coEff2;

			//Potential variables.
			double a1;
			double a2;
			double a3;

		public:

			/**
			 * @brief Creates an new AO Potential.
			 * @param cfg The address of the configuration file reader.
			 */
			AOPotential(configReader::config* cfg);
			/**
			 * @brief Releases the force from memory.
			 */
			~AOPotential();

			/**
			 * @brief Get the force from the AO Potential.
			 * @param index The index particle to calculated the force on.
			 * @param nPart The number of particles in the system.
			 * @param boxSize The size of the system.
			 * @param time The current system time.
			 * @param itemCell The cell containing the index particle.
			 * @param items All particles in the system.
			 */
			void getAcceleration(int index, int nPart, int boxSize, double time, simulation::cell* itemCell, simulation::particle** items);
			/**
			 * @brief Flag for a force dependent time.
			 * @return True for time dependent. False otherwise. 
			 */
			bool isTimeDependent() { return false; }
			/**
			 * @brief Checks for particle interation between the index particle and all particles in the provided cell.
			 * @param boxSize The size of the system.
			 * @param time The current system time.
			 * @param index The particle to find the force on.
			 * @param itemCell The cell to check for interactions in.
			 */
			void iterCells(int boxSize, double time, simulation::particle* index, simulation::cell* itemCell);

	};

}

#endif // FORCE_H
