#include "config.h"

namespace configReader
{

config::config(string fileName)
{
	//Open the config file.
	ifstream cfg;
	cfg.open(fileName, std::ios_base::in);

	for (std::string line; getline(cfg,line); )
	{

		if(line.at(0) != '#')
		{
			//Get the position of the equal sign.
			int pos = line.find("=");

			//Add the left of the equal sign to the key.
			string key = line.substr(0, pos);

			//Remove any leading or trailing white space.

			//No Regex in g++ 4.8 :'(
			while (std::isspace( *key.begin() ))
			{
				key.erase(key.begin());
			}
			while (std::isspace( *key.rbegin() ))
			{
				key.erase(key.length()-1);
			}
			//key = regex_replace(key, std::regex("(^\\s+)"), "");
			//key = regex_replace(key, std::regex("(\\s+$)"), "");

			//Add the right of the equal sign to the value.
			line.erase(0,pos+1);
			string val = line;

			while (std::isspace( *val.begin() ))
			{
				val.erase(val.begin());
			}
			while (std::isspace( *val.rbegin() ))
			{
				val.erase(val.length()-1);
			}
			//val = regex_replace(val, std::regex("(^\\s+)"), "");
			//val = regex_replace(val, std::regex("(\\s+$)"), "");

			options[key] = val;
		}
	}

}

config::~config()
{
}

template<typename T> T config::getParam(string key)
{
	return dynamic_cast<T>(options[key]);
}

template<> int config::getParam<int>(string key)
{
	return std::stoi(options[key],nullptr);
}

template<> float config::getParam<float>(string key)
{
	return std::stof(options[key],nullptr);
}

template<> double config::getParam<double>(string key)
{
	return std::stod(options[key],nullptr);
}

template<> string config::getParam<string>(string key)
{
	return options[key];
}

}

