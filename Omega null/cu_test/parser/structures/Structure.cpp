#include"../../global_includes.h"




namespace on {



	Structure::Structure(Structure* parent) {
		_parent = parent;
		_data = parent->_data;
	}


}
