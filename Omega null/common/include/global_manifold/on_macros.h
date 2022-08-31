#pragma once


//#define __kernel__(_xml_object_name_, _content_) XMLObject _xml_object_name_ << (#_content_)

inline int now_ms() { return std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count(); }

#define IGNORE_DIM 1

#define ON_STRING(_content_) #_content_

//a struct which contains only data
#define On_Being struct

//a static struct which contains only functions
#define On_Process static struct

//a namespace which contains both data and functions
#define On_Structure namespace

#define On_Sync(_name_)\
cudaDeviceSynchronize();\
on::Debug::cuda_error = cudaGetLastError();\
if(on::Debug::cuda_error != cudaSuccess){\
	std::cout << std::endl << "CUDA ERROR AT " << #_name_ << ": " << cudaGetErrorName(on::Debug::cuda_error) << ", " << cudaGetErrorString(on::Debug::cuda_error) << std::endl;\
}


#define On_Copy(_to_, _from_, _length_)\
 __pragma(unroll) for (int i = 0; i < _length_; i++){\
	_to_[i] = _from_[i];\
 }




namespace on {
	enum host_or_device { //find a better name
		host = 0,
		device = 1,
	};

	enum direction {
		host_to_device = 0,
		device_to_host = 1,
	};
}




