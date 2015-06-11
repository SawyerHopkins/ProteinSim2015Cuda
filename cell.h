#ifndef CELL_H
#define CELL_H
#include <vector>
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

			//Add member particle.
			void addMember(particle* item) { members.push_back(item); item->setIndex(members.size() - 1); }

			//Remove member particle
			void removeMember(int index);

	};

}

#endif // CELL_H
