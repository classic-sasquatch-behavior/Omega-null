
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

	af::array test_array(10, 10, s32);
	test_array = 5;
	on::Tensor<int> test_tensor;

	//test array to tensor
	test_tensor = test_array;

	//test tensor to array



	//print array
	on::Debug::Print::af_array("test array", test_array);

	//print tensor
	on::Debug::Print::tensor(test_tensor);

	on::Debug::wait();
	return 0;
}
