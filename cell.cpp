#include "cell.h"

namespace simulation
{

	cell::cell()
	{
	}

	cell::~cell()
	{
	}

	void cell::removeMember(int index)
	{
		members.erase(members.begin() + index); 
		if (members.size() != members.capacity()) 
		{
			members.shrink_to_fit();
		}
	}

}

