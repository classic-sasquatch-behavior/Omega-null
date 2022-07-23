#include"../../headers/global_includes.h"




namespace on {



	Structure::Structure(Structure* parent, int level) {
		_parent = parent;
		//_data = parent->get_data();
		_level = level;
	}


}
