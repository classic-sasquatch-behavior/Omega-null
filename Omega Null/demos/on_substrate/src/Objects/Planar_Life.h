#pragma once
#include"substrate_manifold.h"
#include"omega_null.h"
#include"omega_null/display.h"

__global__ void change_environment(on::Tensor<int> environment, on::Tensor<int> cells) {
	GET_DIMS(maj, min);
	CHECK_BOUNDS(environment.maj_span, environment.min_span);

	int affect = cells(maj, min, 0) * cells(maj, min, 1);
	FOR_3X3_INCLUSIVE(n_maj, n_min, environment.maj_span, environment.min_span, maj, min, 
		atomicAdd(&environment(maj, min), affect);
	);
}

__global__ void move(on::Tensor<int> environment, on::Tensor<int> cells, on::Tensor<int> future_cells) {
	GET_DIMS(maj, min);
	CHECK_BOUNDS(environment.maj_span, environment.min_span);

	int attractor = cells(maj, min, 1);
	int self_attraction = environment(maj, min) * attractor;
	
	int highest_value = self_attraction;
	int largest_neighbor_maj = maj;
	int largest_neighbor_min = min;

	FOR_3X3_INCLUSIVE(n_maj, n_min, environment.maj_span, environment.min_span, maj, min, 
		int target_value = environment(n_maj, n_min) * attractor;
		if (target_value > highest_value) {
			highest_value = target_value;
			largest_neighbor_maj = n_maj;
			largest_neighbor_min = n_min;
		}
	);

	atomicAdd(&future_cells(largest_neighbor_maj, largest_neighbor_min, 0), cells(maj, min, 0));
	atomicAdd(&future_cells(largest_neighbor_maj, largest_neighbor_min, 1), attractor);

}

//pretty goofy way to do this to be honest. But let's see how slow it is.
__global__ void draw(on::Tensor<int> input, on::Tensor<uchar> output) {
	GET_DIMS(maj, min);
	CHECK_BOUNDS(input.maj_span, input.min_span);

	int value = input(maj, min, 0);
	int attractor = input(maj, min, 1);

	uchar color[3];

	int first_parity = (input(min, maj, 0) > 0); 
	int second_parity = (input(min, maj, 1) > 0);
	
	int first_zero = value != 0;
	int second_zero = attractor != 0;

	int zeroes = first_zero + second_zero;


	switch (first_parity) {
		case 0: switch (second_parity) {
			case 0: uchar color[3] = {0, 0, 255}; break;
			case 1: uchar color[3] = {255, 0, 255}; break;
		} break;
		case 1: switch (second_parity) {
			case 0: uchar color[3] = {255, 255, 0}; break;
			case 1: uchar color[3] = {255, 0, 0}; break;
		} break;
	}

	for (int channel = 0; channel < 3; channel++) { 
		int channel_value = color[channel];
		int new_value = (channel_value * zeroes) / 2;
		color[channel] = new_value;
		output(maj, min, channel) = color[channel];
	}
}




namespace on {

	On_Structure Substrate {

		On_Structure Species {

			On_Structure Planar_Life {

				On_Process Draw {
					static on::Tensor<uchar> frame(on::Tensor<int>& cells) {

					on::Tensor<uchar> output ({cells.maj_span, cells.min_span, 3}, 0);

					on::Launch::Kernel::conf_2d(cells.maj_span, cells.min_span);
					draw<<<LAUNCH>>(cells, output);
					On_Sync(draw);

					return output;

					}
				};
				
				On_Structure Parameter {
					static bool running = false;
					const int environment_width = 512;
					const int environment_height = 512;
				}

				static void step(Tensor<int>& environment, Tensor<int>& cells) {

					on::Launch::Kernel::conf_2d(environment.maj_span, environment.min_span);

					change_environment<<<LAUNCH>>>(environment, cells);
					On_Sync(change_environment);

					on::Tensor<int> future_cells({cells.spans[0], cells.spans[1], cells.spans[2]}, 0);

					move<<<LAUNCH>>>(environment, cells, future_cells);
					On_Sync(move);

					cells = future_cells;

				}	

				static void run() {

					Display::Forge::Initialize::window();
					Planar_Life::Parameter::running = true;

					on::Tensor<int> environment;
					on::Tensor<int> cells; //channel 0: values //channel 1: attractors

					on::Tensor<uchar> frame({Parameter::environment_width, Parameter::environment_height}, 3);

					//step:

					do {
						Display::Forge::Listen::for_input();
						step(environment, cells);
						frame = Draw::frame(cells); //probably want to make the whole frame drawing system static somehow
						Display::Forge::render(frame);
					} while (Planar_Life::Parameter::running);







				}
			}
		}
	}
}






