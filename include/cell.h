#ifndef CELL_H
#define CELL_H
#include <map>
#include "particle.h"

namespace simulation
{

	/**
	 * @class cell
	 * @author Sawyer Hopkins
	 * @date 06/27/15
	 * @file cell.h
	 * @brief Cells for the particle grid.
	 */
	class cell
	{

		private:

			//The particles in the cell.
			std::map<int,particle*> members;

		public:

			//Header Version.
			const int version = 1;

			//Constructor and Destructor
			cell();
			~cell();

			//Cell neighbors
			cell* top;
			cell* bot;
			cell* left;
			cell* right;
			cell* front;
			cell* back;

			//Member management.
			/**
			 * @brief Adds a particle to the cell manager.
			 * @param p The address of the particle to add.
			 */
			void addMember(particle* p);
			/**
			 * @brief Removes a particle from the cell manager.
			 * @param p The address of the particle to remove.
			 */
			void removeMember(particle* p);
			/**
			 * @brief Expose the pointer to the first particle in the cell.
			 * @return 
			 */
			const std::map<int,particle*>::iterator getBegin() { return members.begin(); }
			/**
			 * @brief Expose the end of the bucket.
			 * @return 
			 */
			const std::map<int,particle*>::iterator getEnd() { return members.end(); }
			/**
			 * @brief Return a specific member by name.
			 * @param key The name of the particle to get.
			 * @return 
			 */
			const std::map<int,particle*>::mapped_type getMember(int key) { return members[key]; }

	};

}

#endif // CELL_H
