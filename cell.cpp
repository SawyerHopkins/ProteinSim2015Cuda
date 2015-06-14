#include "cell.h"

namespace simulation
{

	cell::cell()
	{
	}

	cell::~cell()
	{
		for (std::vector<particle*>::iterator i = members.begin(); i != members.end(); i++)
		{
			delete[] &i;
		}
		delete[] &members;
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

