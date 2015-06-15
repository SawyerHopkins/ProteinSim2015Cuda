#ifndef CELL_H
#define CELL_H
#include "particle.h"

namespace simulation
{

	class cell
	{

		private:
			std::vector<particle*> members;

		public:

			//Constructor/Destructor
			cell();
			~cell();

			//Neighbor cell references.
			cell* left;
			cell* right;
			cell* top;
			cell* bot;
			cell* front;
			cell* back;

			/**
			 * @brief Adds a particle to the cell.
			 * @param item The particle to add.
			 */
			void addMember(particle* item) { members.push_back(item); item->setIndex(members.size() - 1); }

			/**
			 * @brief Removes a particle from the cell.
			 * @param index The index of the particle to remove.
			 */
			void removeMember(int index);
			particle* getMember( int index ) const { return members[index]; }

			int memberCount() { return members.size(); }

	};

}

#endif // CELL_H
