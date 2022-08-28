
#include"omega_null.h"


	/****************************************OMEGA NULL****************************************
	This repository is a series of projects which develop the Omega Null language and library.
	There are four projects, two of which comprise the useful part of the library, intended to
	be used in other projects. The other two are demo projects, intended to provide a testing
	ground for the library as it develops. To be specific:
	
	on_language: a framework for describing kernels, and code for the associated precompiler. //usable
	
	on_library: a matrix manipulation library designed for interoperability with other libraries and CUDA //in progress
	
	on_substrate: cellular automaton demonstration //on the backburner
	
	on_vision: computer vision demonstration //SLIC in progress

	*/


#pragma region run

	using namespace on;
	using namespace Vision::Meta;
	int main() {

		Vision::Clip<int> source;
		Vision::Clip<int> SLIC;
		Vision::Load::clip(source, Parameter::data_path);

		Vision::Algorithm::SLIC::run(source, SLIC);

		Vision::Window::Display::clip(SLIC);
		on::Debug::wait(); //conflict here between on::wait and opencv equivalent

		return 0;
	}

#pragma endregion