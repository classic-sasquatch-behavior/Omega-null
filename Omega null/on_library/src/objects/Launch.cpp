#include"global_manifold.h"
#include"on_library.h"




namespace on {
	void Launch::Kernel::conf_2d(int maj_span, int min_span) {
		Launch::Parameter::block_dim_x = 32;
		Launch::Parameter::block_dim_y = 32;

		
		Launch::Parameter::grid_dim_x = (maj_span - (maj_span % Launch::Parameter::block_dim_x)) / Launch::Parameter::block_dim_x;
		Launch::Parameter::grid_dim_y = (min_span - (min_span % Launch::Parameter::block_dim_y)) / Launch::Parameter::block_dim_y;

		Launch::Parameter::num_blocks = { Launch::Parameter::grid_dim_x, Launch::Parameter::grid_dim_y, 1 };
		Launch::Parameter::threads_per_block = { Launch::Parameter::block_dim_x, Launch::Parameter::block_dim_y, 1 };
	}



	void Launch::Kernel::conf_1d(int maj_span) {
		Launch::Parameter::block_dim_x = 1024;

		Launch::Parameter::grid_dim_x = (maj_span - (maj_span % Launch::Parameter::block_dim_x)) / Launch::Parameter::block_dim_x;

		Launch::Parameter::num_blocks = { Launch::Parameter::grid_dim_x, 1, 1 };
		Launch::Parameter::threads_per_block = { Launch::Parameter::block_dim_x, 1, 1 };
	}

}


