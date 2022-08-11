#pragma once

#include<vector_types.h>

typedef unsigned int uint;
typedef unsigned char uchar;
typedef void kernel;



//aquire the coordinates of the thread. works on 2d kernels as well as 1d, if youre okay with ignoring one of the dimensions.
#define GET_DIMS(_maj_dim_, _min_dim_)							\
	int _maj_dim_ = (blockIdx.x * blockDim.x) + threadIdx.x;	\
	int _min_dim_ = (blockIdx.y * blockDim.y) + threadIdx.y;	\
	int & _MAJ_ = _maj_dim_;									\
	int & _MIN_ = _min_dim_;										

//check if the thread is within the bounds of the d_Mat given as shape, and if not, return the thread.
#define CHECK_BOUNDS(_max_maj_, _max_min_) if((_MAJ_ < 0)||(_MIN_ < 0)||(_MAJ_ >= _max_maj_)||(_MIN_ >= _max_min_)){return;} 

//casts a larger dimension to a smaller one
#define CAST_DOWN(_old_coord_, _new_max_) \
	((_old_coord_ - (_old_coord_ % _new_max_ ))/ _new_max_)

//casts a smaller dimension to a larger one. tries to place the new coordinate in the middle of each 'segment'
#define CAST_UP(_old_coord_, _old_max_, _new_max_) \
	((_old_coord_*(_new_max_/_old_max_))+(((_new_max_/_old_max_)-((_new_max_/_old_max_)%2))/2))

//iterates through the elements directly adjacent to the given coordinates in a 2d plane, excluding self
#define FOR_NEIGHBOR(_new_maj_, _new_min_, _maj_max_, _min_max_, _origin_maj_, _origin_min_, _content_)	\
	__pragma(unroll) for (int _neighbor_maj_ = -1; _neighbor_maj_ < 2; _neighbor_maj_++) {				\
		__pragma(unroll) for (int _neighbor_min_ = -1; _neighbor_min_ < 2; _neighbor_min_++) {			\
			int _new_maj_ = _origin_maj_ + _neighbor_maj_;											    \
			int _new_min_ = _origin_min_ + _neighbor_min_;												\
			if((_new_maj_ < 0)||(_new_min_ < 0)||(_new_maj_ >= _maj_max_)||(_new_min_ >= _min_max_ )	\
							  ||((_new_maj_ == _origin_maj_)&&(_new_min_ == _origin_min_))) {continue;} \
			_content_;																				    \
		}																							    \
	}

//virtually transform a 2d tensor into a 1d tensor, and return the resulting linear id of the element pointed to by the given coordinates
#define LINEAR_CAST(_maj_dim_, _min_dim_, _min_max_) \
	((_maj_dim_ * _min_max_) + _min_dim_)



#define SYNC_AND_CHECK_FOR_ERRORS(_kernel_)											 \
{																					 \
	cudaDeviceSynchronize();														 \
	cudaError_t error = cudaGetLastError();											 \
	if(error != cudaSuccess) {														 \
		std::cout << "Error in kernel " << #_kernel_								 \
		<< " at " << __FILE__ << ":" << __LINE__ << ": "							 \
		<< cudaGetErrorName(error) << ":" << cudaGetErrorString(error) << std::endl; \
		abort();													  				 \
		}																			 \
}		



#define ON_CUDA_ERROR_CHECK(_where_) { \
cudaError_t error = cudaGetLastError(); if(error != cudaSuccess) std::cout << std::endl << "CUDA error " << cudaGetErrorName(error) \
<< " at " << #_where_ <<": " << cudaGetErrorString(error) << std::endl;}












//inline functions

namespace on {
	static class Kernel {
	public:
		Kernel();
		static uint _block_dim_x_;
		static uint _block_dim_y_;
		
		static uint _grid_dim_x_;
		static uint _grid_dim_y_;

		static dim3 _num_blocks_;
		static dim3 _threads_per_block_;

		static void conf_2d(int maj_span, int min_span);
		static void conf_1d(int maj_span);

		static dim3 num_blocks() { return _num_blocks_; }
		static dim3 threads_per_block() { return _threads_per_block_; }
	};
}

#define KERNEL_SHAPE on::Kernel::num_blocks(), on::Kernel::threads_per_block()
