#pragma once
#include"global_manifold.h"

namespace on {
	On_Structure Launch{
		On_Structure Parameter {
			static uint block_dim_x = 0;
			static uint block_dim_y = 0;

			static uint grid_dim_x = 0;
			static uint grid_dim_y = 0;

			static dim3 num_blocks(0,0,0);
			static dim3 threads_per_block(0,0,0);
		}

		On_Process Kernel {
			static void conf_2d(int maj_span, int min_span);
			static void conf_1d(int maj_span);
		};
	}

	#define LAUNCH on::Launch::Parameter::num_blocks, on::Launch::Parameter::threads_per_block
}
