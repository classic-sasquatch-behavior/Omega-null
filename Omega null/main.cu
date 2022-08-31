
#include"omega_null.h"
#include"omega_null/vision.h"
#include"omega_null/signals.h"
#include"omega_null/substrate.h"


	/****************************************OMEGA NULL****************************************
	This repository is a series of projects which develop the Omega Null language and library.
	There are four projects, two of which comprise the useful part of the library, intended to
	be used in other projects. The other two are demo projects, intended to provide a testing
	ground for the library as it develops. To be specific:
	
	on_language: a framework for describing kernels, and code for the associated precompiler. //usable
	
	on_library: a matrix manipulation library designed for interoperability with other libraries and CUDA //in progress

	on_display: window and display backend //just added
	
	on_substrate: cellular automaton demonstration //added Planar_Life, needs testing
	
	on_vision: computer vision demonstration //SLIC in progress

	on_signals: signal processing library //just added

	*/

#define TESTING_SUBSTRATE


#pragma region run

	#ifdef TESTING_VISION

		using namespace on;
		using namespace Vision::Meta;
		int main() {

			Vision::Clip<int> source;
			Vision::Clip<int> SLIC;
			Vision::Load::clip(source, Parameter::data_path);

			Vision::Algorithm::SLIC::run(source, SLIC);

			Vision::Window::Display::clip(SLIC); //move to Display::OpenCV
			on::Debug::wait(); //potential conflict here between on::wait and opencv equivalent

			return 0;
		}

	#endif


	#ifdef TESTING_SUBSTRATE
		
		using namespace on;
		using On_Structure Substrate::Species;
		int main() {
			srand(time(NULL));

			on::Tensor<int> seed = Planar_Life::Seed::cells(rand());

			Planar_Life::run(seed);

			return 0;
		}

	#endif






#pragma endregion