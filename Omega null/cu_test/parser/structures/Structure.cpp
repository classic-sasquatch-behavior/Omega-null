#include"../../global_includes.h"




namespace on {



	Structure::Structure(Structure* parent, int level) {
		_parent = parent;
		_data = parent->_data;
		_level = level;
	}


}
