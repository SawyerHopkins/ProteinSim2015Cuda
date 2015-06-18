#include "cell.h"

namespace simulation
{

	cell::cell()
	{
		//Set neighbors as self until initialized by the system.
		top = this;
		bot = this;
		front = this;
		back = this;
		left = this;
		right = this;
	}

	cell::~cell()
	{
		delete[] &members;
	}

	/**
	 * @brief Adds a particle to the cell manager.
	 * @param p The address of the particle to add.
	 */
	void cell::addMember(particle* p)
	{
		//Adds member with name as the key.
		members[p->getName()] = p;
	}

	/**
	 * @brief Removes a particle from the cell manager.
	 * @param p The address of the particle to remove.
	 */
	void cell::removeMember(particle* p)
	{
		//Removes the member by associated key.
		members.erase(p->getName());
	}

}

