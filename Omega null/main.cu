
#include"omega_null.h"


	/****************************************OMEGA NULL****************************************
	This repository is a series of projects which develop the Omega Null language and library.
	There are four projects, two of which comprise the useful part of the library, intended to
	be used in other projects. The other two are demo projects, intended to provide a testing
	ground for the library as it develops. To be specific:

	on_language: a framework for describing kernels, and code for the associated precompiler.

	on_library: a matrix manipulation library designed for interoperability with other libraries and CUDA

	on_substrate: cellular automaton demonstration

	on_vision: computer vision demonstration
	*/

int main() {

	cv::Mat test_mat(cv::Size{10,10}, CV_32SC1, 5);
	on::Tensor<int> test_tensor({10,10}, 7);
	cv::cuda::GpuMat test_gpu_mat; //0
	test_gpu_mat.upload(test_mat);

	test_gpu_mat = test_tensor;

	on::Debug::Print::d_Mat<int>(test_gpu_mat);
	on::Debug::Print::h_Mat<int>(test_mat);
	on::Debug::Print::tensor(test_tensor);
	on::Debug::wait();
	return 0;
}
