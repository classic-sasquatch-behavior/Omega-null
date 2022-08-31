#pragma once
#include"substrate_manifold.h"
#include"omega_null.h"
#include"omega_null/display.h"






namespace on {

	On_Structure Substrate {

		On_Structure Species {

			On_Structure Planar_Life {

				On_Process Seed {
					static on::Tensor<int> cells(int value);
				};

				On_Process Draw {
					static on::Tensor<uchar> frame(on::Tensor<int>& cells);
				};
				
				On_Structure Parameter {
					static bool running = false;
					const int environment_width = 512;
					const int environment_height = 512;
				}

				On_Process Step {
					static void polar(Tensor<int>& environment, Tensor<int>& cells);
				};


				static void run(on::Tensor<int> seed = Planar_Life::Seed::cells(rand())) {

					//Display::Forge::Initialize::window();
					Planar_Life::Parameter::running = true;

					on::Tensor<int> environment({Parameter::environment_width, Parameter::environment_height},0);
					on::Tensor<int> cells = seed; //channel 0: values //channel 1: attractors

					on::Tensor<uchar> frame({Parameter::environment_width, Parameter::environment_height}, 3);

					do {
						//Display::Forge::Listen::for_input();
						Step::polar(environment, cells);
						frame = Draw::frame(cells); //probably want to make the whole frame drawing system static somehow
						Display::Forge::render(frame);
					} while (Planar_Life::Parameter::running);

				}
			}
		}
	}
}






