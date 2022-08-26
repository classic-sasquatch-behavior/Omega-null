#pragma once

#include<vector_types.h>

typedef unsigned int uint;
typedef unsigned char uchar;

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

#define FOR_MXN_INCLUSIVE(_new_maj_, _new_min_, _M_, _N_, _maj_max_, _min_max_, _origin_maj_, _origin_min_, ...)				 \
int _maj_limit_ = (_M_ - (_M_ % 2)) / 2;																			   			 \
int _min_limit_ = (_N_ - (_N_ % 2)) / 2;																						 \
	__pragma(unroll) for (int _neighbor_maj_ = -_maj_limit_; _neighbor_maj_ < (_maj_limit_ + (_M_ % 2)); _neighbor_maj_++) {	 \
		__pragma(unroll) for (int _neighbor_min_ = -_min_limit_; _neighbor_min_ < (_min_limit_ + (_N_ % 2)); _neighbor_min_++) { \
			int _new_maj_ = _origin_maj_ + _neighbor_maj_;																		 \
			int _new_min_ = _origin_min_ + _neighbor_min_;																	     \
			if((_new_maj_ < 0)||(_new_min_ < 0)||(_new_maj_ >= _maj_max_)||(_new_min_ >= _min_max_ )){continue;}				 \
			__VA_ARGS__;																										 \
		}																														 \
	}

#define FOR_MXN_EXCLUSIVE(_new_maj_, _new_min_, _M_, _N_, _maj_max_, _min_max_, _origin_maj_, _origin_min_, ...)				 \
int _maj_limit_ = (_M_ - (_M_ % 2)) / 2;																						 \
int _min_limit_ = (_N_ - (_N_ % 2)) / 2;																						 \
	__pragma(unroll) for (int _neighbor_maj_ = -_maj_limit_; _neighbor_maj_ < (_maj_limit_ + (_M_ % 2)); _neighbor_maj_++) {	 \
		__pragma(unroll) for (int _neighbor_min_ = -_min_limit_; _neighbor_min_ < (_min_limit_ + (_N_ % 2)); _neighbor_min_++) { \
			int _new_maj_ = _origin_maj_ + _neighbor_maj_;																		 \
			int _new_min_ = _origin_min_ + _neighbor_min_;																		 \
			if((_new_maj_ < 0)||(_new_min_ < 0)||(_new_maj_ >= _maj_max_)||(_new_min_ >= _min_max_ )							 \
							  ||((_new_maj_ == _origin_maj_)&&(_new_min_ == _origin_min_))) {continue;}							 \
			__VA_ARGS__;																										 \
		}																														 \
	}

	#define FOR_3X3_INCLUSIVE(_new_maj_, _new_min_, _maj_max_, _min_max_, _origin_maj_, _origin_min_, ...) \
	FOR_MXN_INCLUSIVE(_new_maj_, _new_min_, 3, 3, _maj_max_, _min_max_, _origin_maj_, _origin_min_, __VA_ARGS__)

	#define FOR_NEIGHBOR(_new_maj_, _new_min_, _maj_max_, _min_max_, _origin_maj_, _origin_min_, ...) \
	FOR_MXN_EXCLUSIVE(_new_maj_, _new_min_, 3, 3, _maj_max_, _min_max_, _origin_max_, _origin_min_, __CA_ARGS__)


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

