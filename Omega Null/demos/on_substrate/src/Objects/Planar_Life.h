#pragma once
#include"substrate_manifold.h"
#include"omega_null.h"

__global__ void change_environment() {

}

__global__ void move() {

}



namespace on {

	On_Structure Substrate {

		On_Structure Species {

			On_Structure Planar_Life {
				
				On_Structure Parameter {
				static bool running = false;
				}

				static void step(Tensor<int>& environment, Tensor<int>& cells) {
					on::Launch::Kernel::conf_2d(environment.maj_span, environment.min_span);

					change_environment<<<LAUNCH>>>(environment, cells);
					On_Sync(change_environment);

					move<<<LAUNCH>>>(environment, cells);
					On_Sync(move);

				}	

				static void run() {
					Planar_Life::Parameter::running = true;

					on::Tensor<int> environment;
					on::Tensor<int> cells; //channel 0: values //channel 1: attractors

					std::vector<on::Tensor<int>> working_frame;
					//step:

					do {
						Listen::for_input();
						step(environment, cells);
						Frame new_frame = Draw::frame(working_frame); //probably want to make the whole frame drawing system static somehow
						Display::frame(new_frame);
					} while (Planar_Life::Parameter::running);







				}
			}
		}
	}
}






