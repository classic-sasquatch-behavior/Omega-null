#pragma once
#include"../../global_includes.h"


namespace on {



	struct Structure {
	public:
		Structure(Structure* parent);
		~Structure();

	private:
		int _num_dims;
		std::string _type;
		std::string _name;
		std::string _data;
		std::vector<std::string> _dim_names;
		Structure* _parent = nullptr;
		std::vector<Structure*> _children;
		std::vector<std::string> _content;
	};

}