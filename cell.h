#ifndef CELL_H
#define CELL_H
#include <map>
#include "particle.h"

namespace simulation
{

	class cell
	{

		private:

			//The particles in the cell.
			std::map<int,particle*> members;

		public:

			//Constructor and Destructor
			cell();
			~cell();

			cell* top;
			cell* bot;
			cell* left;
			cell* right;
			cell* front;
			cell* back;

			//Member management.
			void addMember(particle* p);
			void removeMember(particle* p);
			const std::map<int,particle*>::iterator getBegin() { return members.begin(); }
			const std::map<int,particle*>::iterator getEnd() { return members.end(); }
			const std::map<int,particle*>::mapped_type getMember(int key) { return members[key]; }

	};

}

#endif // CELL_H
