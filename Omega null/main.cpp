#include"global_config.h"
#include"omega_null.h"


	/***********************************OMEGA NULL*******************************************
	This repository is a series of projects which develop the Omega Null language and library.
	There are four projects, two of which comprise the useful part of the library, intended to
	be used in other projects. The other two are demo projects, intended to provide a testing
	ground for the library as it develops. To be specific:

	on_language: a framework for describing kernels, and code for the associated precompiler.

	on_library: a matrix storage library designed for easy interoperability

	on_substrate: cellular automaton demonstration

	on_vision: computer vision demonstration
	*/

int main() {

	#ifdef ON_LIBRARY
		on::Library::main();
	#endif

	#ifdef ON_LANGUAGE
		on::Language::main();
	#endif

	#ifdef ON_SUBSTRATE
		on::Substrate::main();
	#endif

	#ifdef ON_VISION
		on::Vision::main();
	#endif

	//write test code here
	{











	}

	return 0;
}
