#pragma once
#include"global_manifold.h"

namespace on {
	On_Structure Launch{
		On_Structure Parameter {
			inline uint block_dim_x = 0;
			inline uint block_dim_y = 0;

			inline uint grid_dim_x = 0;
			inline uint grid_dim_y = 0;

			inline dim3 num_blocks(0,0,0);
			inline dim3 threads_per_block(0,0,0);
		}

		On_Process Kernel {

			static void conf_2d(int maj_span, int min_span) {
				Launch::Parameter::block_dim_x = 32;
				Launch::Parameter::block_dim_y = 32;

				Launch::Parameter::grid_dim_x = (maj_span - (maj_span % Launch::Parameter::block_dim_x)) / Launch::Parameter::block_dim_x;
				Launch::Parameter::grid_dim_y = (min_span - (min_span % Launch::Parameter::block_dim_y)) / Launch::Parameter::block_dim_y;

				Launch::Parameter::num_blocks = { Launch::Parameter::grid_dim_x + 1, Launch::Parameter::grid_dim_y + 1, 1 };
				Launch::Parameter::threads_per_block = { Launch::Parameter::block_dim_x, Launch::Parameter::block_dim_y, 1 };
			}
 
			static void conf_1d(int maj_span) {
				Launch::Parameter::block_dim_x = 1024;

				Launch::Parameter::grid_dim_x = (maj_span - (maj_span % Launch::Parameter::block_dim_x)) / Launch::Parameter::block_dim_x;

				Launch::Parameter::num_blocks = { Launch::Parameter::grid_dim_x + 1, 1, 1 };
				Launch::Parameter::threads_per_block = { Launch::Parameter::block_dim_x, 1, 1 };
			}
		};

	}

	#define LAUNCH on::Launch::Parameter::num_blocks, on::Launch::Parameter::threads_per_block


}
