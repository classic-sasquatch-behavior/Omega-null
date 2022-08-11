#pragma once


namespace on {
	ON_STRUCTURE Launch {
		static ON_BEING Parameters {
			static uint block_dim_x;
			static uint block_dim_y;

			static uint grid_dim_x;
			static uint grid_dim_y;

			static dim3 num_blocks;
			static dim3 threads_per_block;
		}

		ON_PROCESS Kernel {
			static void conf_2d(int maj_span, int min_span);
			static void conf_1d(int maj_span);
		}




	};
}
