#include "cell.h"

namespace simulation
{

	cell::cell()
	{
		top = this;
		bot = this;
		front = this;
		back = this;
		left = this;
		right = this;
	}

	cell::~cell()
	{
	}

	void cell::addMember(particle* p)
	{
		members[p->getName()] = p;
	}

	void cell::removeMember(particle* p)
	{
		members.erase(p->getName());
	}

}

