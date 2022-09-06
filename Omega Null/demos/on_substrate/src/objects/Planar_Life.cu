
#include"substrate_manifold.h"
#include"omega_null.h"
#include"Planar_Life.h"


__global__ void change_environment(on::Device_Ptr<int> environment, on::Device_Ptr<int> cells) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());

	int affect = cells(maj, min, 0) * cells(maj, min, 1);
	FOR_3X3_INCLUSIVE(n_maj, n_min, environment.maj(), environment.min(), maj, min,
		atomicAdd(&environment(maj, min), affect);
	);
}

__global__ void move(on::Device_Ptr<int> environment, on::Device_Ptr<int> cells, on::Device_Ptr<int> future_cells) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());

	int attractor = cells(maj, min, 1);
	int self_attraction = environment(maj, min) * attractor;

	int highest_value = self_attraction;
	int largest_neighbor_maj = maj;
	int largest_neighbor_min = min;

	FOR_3X3_INCLUSIVE(n_maj, n_min, environment.maj(), environment.min(), maj, min,
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
__global__ void draw(on::Device_Ptr<int> input, on::Device_Ptr<uchar> output) {
	DIMS_2D(maj, min);
	BOUNDS_2D(input.maj(), input.min());

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

__global__ void spawn(on::Device_Ptr<int> mask, curandState* random_states, on::Device_Ptr<int> result) {
	DIMS_2D(maj, min);
	BOUNDS_2D(result.maj(), result.min());
	//if (mask(maj, min) == 0) { return; } //possible failure here

	int id = LINEAR_CAST(maj, min, result.min());

	curandState local_state = random_states[id]; //possible failure here (unlikely)
	int random_0 = curand(&local_state);
	int random_1 = curand(&local_state);

	int cell_value = (random_0 % 20) - 10;
	int attractor_value = (2*(random_1 % 2)) - 1;
	
	result(maj, min, 0) = cell_value; //possible failure here
	result(maj, min, 1) = attractor_value; //possible failure here
}

namespace on {

	On_Structure Substrate {

		On_Structure Species {

			On_Structure Planar_Life {

				on::Tensor<int> Seed::cells(int value = 0) {

					on::Tensor<int> result({Parameter::environment_width, Parameter::environment_height, 2}, 0, "result");
					af::array af_mask = (af::randu(Parameter::environment_width, Parameter::environment_height) > 0.5).as(s32);
					on::Tensor<int> mask({Parameter::environment_width, Parameter::environment_height}, 0, "mask");
					//mask = af_mask; //possible failure here

					curandState* states = on::Random::Initialize::curand_xor(Parameter::environment_area, value);

					on::configure::kernel_2d(result.maj(), result.min());
					spawn<<<LAUNCH>>>(mask, states, result); //try using the nvidia debugger
					SYNC_KERNEL(spawn); 

					cudaFree(states); //bad way of doing this, because it's not clear that one would have to call cudafree on curand_xor. should at least put it in on::Random::Delete

					return result;

				}

				on::Tensor<uchar> Draw::frame(on::Tensor<int>& cells) {

					on::Tensor<uchar> output({cells.maj(), cells.min(), 3}, 0);

					on::configure::kernel_2d(cells.maj(), cells.min());
					draw <<<LAUNCH>>> (cells, output);
					SYNC_KERNEL(draw);

					return output;

				}

				void Planar_Life::Step::polar(Tensor<int>& environment, Tensor<int>& cells) {

					on::configure::kernel_2d(environment.maj(), environment.min());

					change_environment<<<LAUNCH>>> (environment, cells);
					SYNC_KERNEL(change_environment);

					on::Tensor<int> future_cells({cells.spans[0], cells.spans[1], cells.spans[2]}, 0);

					move<<<LAUNCH>>> (environment, cells, future_cells);
					SYNC_KERNEL(move);

					cells = future_cells;

				}
			}
		}
	}
}












