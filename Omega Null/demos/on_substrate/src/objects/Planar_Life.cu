
#include"substrate_manifold.h"
#include"omega_null.h"
#include"Planar_Life.h"


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

//pretty goofy way to do this to be honest. But let's see how fast or slow it runs.
__global__ void draw(on::Tensor<int> input, on::Tensor<uchar> output) {
	GET_DIMS(maj, min);
	CHECK_BOUNDS(input.maj_span, input.min_span);

	int value = input(maj, min, 0);
	int attractor = input(maj, min, 1);

	uchar color[3] = {0,0,0};

	int first_parity = (input(min, maj, 0) > 0);
	int second_parity = (input(min, maj, 1) > 0);

	int first_zero = value != 0;
	int second_zero = attractor != 0;

	int zeroes = first_zero + second_zero;

	const uchar red[3] = {255, 0, 0};
	const uchar green[3] = {0, 255, 0};
	const uchar blue[3] = {0, 0, 255};
	const uchar purple[3] = {255, 0, 255};


	switch (first_parity) {
		case 0: switch (second_parity) {
			case 0: On_Copy(color, blue, 3); break;
			case 1: On_Copy(color, purple, 3); break;
		} break;
		case 1: switch (second_parity) {
			case 0: On_Copy(color, green, 3); break;
			case 1: On_Copy(color, red, 3); break;
		} break;
	}

	for (int channel = 0; channel < 3; channel++) {
		int channel_value = color[channel];
		int new_value = (channel_value * zeroes) / 2;
		color[channel] = new_value;
		output(maj, min, channel) = color[channel];
	}
}

__global__ void spawn(on::Tensor<int> mask, curandState* random_states, on::Tensor<int> result) {
	GET_DIMS(maj, min);
	CHECK_BOUNDS(result.maj_span, result.min_span);
	if (mask(maj, min) == 0) { return; }

	int id = LINEAR_CAST(maj, min, result.min_span);

	curandState local_state = random_states[id];
	int random_0 = curand(&local_state);
	int random_1 = curand(&local_state);

	int cell_value = (random_0 % 20) - 10;
	int attractor_value = (2*(random_1 % 2)) - 1;
	
	result(maj, min, 0) = cell_value;
	result(maj, min, 1) = attractor_value;
}

namespace on {

	On_Structure Substrate {

		On_Structure Species {

			On_Structure Planar_Life {

				on::Tensor<int> Seed::cells(int value = 0) {

					on::Tensor<int> result({Parameter::environment_width, Parameter::environment_height, 3});
					af::array af_mask = (af::randu(Parameter::environment_width, Parameter::environment_height) > 0.5).as(s32);
					on::Tensor<int> mask({Parameter::environment_width, Parameter::environment_height});
					mask = af_mask; //this is likely what's not working


					curandState* states = nullptr;
					on::Random::Initialize::curand_xor(Parameter::environment_area, value, states);

					on::Launch::Kernel::conf_2d(result.maj_span, result.min_span);
					spawn<<<LAUNCH>>>(mask, states, result);
					On_Sync(spawn); 

					cudaFree(states); //bad way of doing this, because it's not clear that one would have to call cudafree on curand_xor. should at least put it in on::Random::Delete

					return result;

				}

				on::Tensor<uchar> Draw::frame(on::Tensor<int>& cells) {

					on::Tensor<uchar> output({cells.maj_span, cells.min_span, 3}, 0);

					on::Launch::Kernel::conf_2d(cells.maj_span, cells.min_span);
					draw <<<LAUNCH>>> (cells, output);
					On_Sync(draw);

					return output;

				}

				void Planar_Life::Step::polar(Tensor<int>& environment, Tensor<int>& cells) {

					on::Launch::Kernel::conf_2d(environment.maj_span, environment.min_span);

					change_environment<<<LAUNCH>>> (environment, cells);
					On_Sync(change_environment);

					on::Tensor<int> future_cells({cells.spans[0], cells.spans[1], cells.spans[2]}, 0);

					move<<<LAUNCH>>> (environment, cells, future_cells);
					On_Sync(move);

					cells = future_cells;

				}
			}
		}
	}
}












