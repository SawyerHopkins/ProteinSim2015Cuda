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
			key = regex_replace(key, std::regex(R"(^\s+)"), "");
			key = regex_replace(key, std::regex(R"(\s+$)"), "");

			//Add the right of the equal sign to the value.
			line.erase(0,pos+1);
			string val = line;

			val = regex_replace(val, std::regex(R"(^\s+)"), "");
			val = regex_replace(val, std::regex(R"(\s+$)"), "");

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

