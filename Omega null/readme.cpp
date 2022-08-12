
#include"global_config.h"


namespace on {
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


}



using namespace on;
int main() {
	#ifdef ON_LANGUAGE
	using ON_STRUCTURE on::Language;
	on::Language::readme();
	#endif

	#ifdef ON_LIBRARY
	using ON_STRUCTURE on::Library;
	on::Library::readme();
	#endif

	#ifdef ON_SUBSTRATE
	using ON_STRUCTURE on::Substrate;
	on::Substrate::readme();
	#endif

	#ifdef ON_VISION
	using ON_STRUCTURE on::Vision;
	on::Vision::readme();
	#endif

	namespace demo_input {

















	}

	return 0;
}
