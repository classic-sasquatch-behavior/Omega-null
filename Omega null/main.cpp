
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

	std::vector<int> temp = { 5,5,5,5,5 };
	std::vector<int> also_temp = { 7,7,7,7,7,7,7 };
	on::Tensor<int> example;

	example = temp;
	also_temp = example;

	on::Debug::Print::tensor(example);

	on::Debug::Print::vector(also_temp);

	on::Debug::wait();

	return 0;
}
