#include<manifold.h>


uint on::Kernel::_block_dim_x_ = 0;
uint on::Kernel::_block_dim_y_ = 0;

uint on::Kernel::_grid_dim_x_ = 0;
uint on::Kernel::_grid_dim_y_ = 0;

dim3 on::Kernel::_num_blocks_ = {0,0,0};
dim3 on::Kernel::_threads_per_block_ = {0,0,0};

on::Kernel::Kernel() {

}

void on::Kernel::conf_2d(int maj_span, int min_span) {
	_block_dim_x_ = 32;
	_block_dim_y_ = 32;

	_grid_dim_x_ = (maj_span - (maj_span % _block_dim_x_)) / _block_dim_x_;
	_grid_dim_y_ = (min_span - (min_span % _block_dim_y_)) / _block_dim_y_;

	_num_blocks_ = { _grid_dim_x_, _grid_dim_y_, 1 };
	_threads_per_block_ = { _block_dim_x_, _block_dim_y_, 1 };
}



void on::Kernel::conf_1d(int maj_span) {
	_block_dim_x_ = 1024;

	_grid_dim_x_ = (maj_span - (maj_span % _block_dim_x_)) / _block_dim_x_;

	_num_blocks_ = { _grid_dim_x_, 1, 1 };
	_threads_per_block_ = { _block_dim_x_, 1, 1 };
}

