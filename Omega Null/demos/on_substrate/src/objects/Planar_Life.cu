
#include"substrate_manifold.h"
#include"omega_null.h"
#include"Planar_Life.h"


//pretty goofy way to do this to be honest. But let's see how fast or slow it runs.
__global__ void draw(sk::Device_Ptr<int> input, sk::Device_Ptr<uchar> output) {
	DIMS_2D(maj, min);
	BOUNDS_2D(input.maj(), input.min());

	int value = input(maj, min, 0);
	int attractor = input(maj, min, 1);

	uchar color[3] = { 50,50,50 };

	int first_parity = (input(min, maj, 0) > 0);
	int second_parity = (input(min, maj, 1) > 0);

	int zero = (value == 0) && (attractor == 0);


	const uchar red[3] = { 255, 100, 100 };
	const uchar green[3] = { 100, 255, 100 };
	const uchar blue[3] = { 100, 100, 255 };
	const uchar purple[3] = { 255, 30, 255 };
	const uchar black[3] = {0,0,0};


	switch (first_parity) {
		case 0: switch (second_parity) {
			case 0: sk_Copy(color, blue, 3); break;
			case 1: sk_Copy(color, purple, 3); break;
		} break;
		case 1: switch (second_parity) {
			case 0: sk_Copy(color, green, 3); break;
			case 1: sk_Copy(color, red, 3); break;
		} break;
	}

	if(zero){ sk_Copy(color, black, 3); }

	for (int channel = 0; channel < 3; channel++) {
		int channel_value = color[channel];
		//int new_value = (channel_value * zeroes) / 2;

		//output(maj, min, channel) = new_value;
		output.device_data[(((channel * input.maj()) + maj ) * input.min()) + min] = channel_value;
	}
}


__global__ void spawn(sk::Device_Ptr<int> mask, curandState* random_states, sk::Device_Ptr<int> result) {
	DIMS_2D(maj, min);
	BOUNDS_2D(result.maj(), result.min());
	//if (mask(maj, min) == 0) { return; } //mask is broken somehow, this seems to delete ALL threads rather than just half.

	int id = LINEAR_CAST(maj, min, result.min());

	curandState local_state = random_states[id];
	int random_0 = curand(&local_state);
	int random_1 = curand(&local_state);

	int cell_value = (random_0 % 20) - 10;
	int attractor_value = (2 * (random_1 % 2)) - 1;

	result(maj, min, 0) = cell_value;
	result(maj, min, 1) = attractor_value;
}

__global__ void change_environment(sk::Device_Ptr<int> environment, sk::Device_Ptr<int> cells) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());

	int affect = cells(maj, min, 0) * cells(maj, min, 1);
	FOR_3X3_INCLUSIVE(n_maj, n_min, environment.maj(), environment.min(), maj, min,
		atomicAdd(&environment(maj, min), affect);
	);
}

__global__ void move(sk::Device_Ptr<int> environment, sk::Device_Ptr<int> cells, sk::Device_Ptr<int> future_cells) {
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

__global__ void dampen_environment(const float damping_factor, sk::Device_Ptr<int> environment) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());

	int value = environment(maj, min);
	environment(maj, min) = value * damping_factor;


}

__global__ void hatch(const int threshold, sk::Device_Ptr<int> cells) {
	DIMS_2D(maj, min);
	BOUNDS_2D(cells.maj(), cells.min());

	int value = cells(maj, min, 0);
	if (value < threshold){return;}

	int attractor = cells(maj, min, 1);
	int value_out = -roundf(value / 8);
	int attractor_out = -fabsf(attractor) / attractor;

	FOR_NEIGHBOR(n_maj, n_min, cells.maj(), cells.min(), maj, min, 
		atomicAdd(&cells(n_maj, n_min, 0), value_out);
		atomicAdd(&cells(n_maj, n_min, 1), attractor_out);
	);
}


namespace on {

	On_Structure Substrate {

		On_Structure Species {

			On_Structure Planar_Life {

				sk::Tensor<int> Seed::cells(int value = 0) {

					sk::Tensor<int> result({Parameter::environment_width, Parameter::environment_height, 2}, 0, "result");
					af::array af_mask = (af::randu(Parameter::environment_width, Parameter::environment_height) > 0.5).as(s32);
					sk::Tensor<int> mask({Parameter::environment_width, Parameter::environment_height}, 0, "mask");
					//mask = af_mask; //possible failure here

					curandState* states = on::Random::Initialize::curand_xor(Parameter::environment_area, value);

					sk::configure::kernel_2d(result.maj(), result.min());
					spawn<<<LAUNCH>>>(mask, states, result); //try using the nvidia debugger
					SYNC_KERNEL(spawn); 

					cudaFree(states); //bad way of doing this, because it's not clear that one would have to call cudafree on curand_xor. should at least put it in on::Random::Delete

					return result;

				}

				sk::Tensor<uchar> Draw::frame(sk::Tensor<int>& cells) {

					sk::Tensor<uchar> output({cells.maj(), cells.min(), 3}, 0);

					sk::configure::kernel_2d(cells.maj(), cells.min());
					draw <<<LAUNCH>>> (cells, output);
					SYNC_KERNEL(draw);

					return output;

				}

				void Planar_Life::Step::polar(sk::Tensor<int>& environment, sk::Tensor<int>& cells) {
					sk::configure::kernel_2d(environment.maj(), environment.min());
					
					//const int threshold = 8;
					//hatch<<<LAUNCH>>>(threshold, cells);
					//SYNC_KERNEL(hatch);

					change_environment<<<LAUNCH>>> (environment, cells); //conversion from tensor to device_ptr
					SYNC_KERNEL(change_environment);

					//const float damping_factor = 0.99;
					//dampen_environment<<<LAUNCH>>>(damping_factor, environment);
					//SYNC_KERNEL(dampen_environment);

					sk::Tensor<int> future_cells({cells.spans[0], cells.spans[1], cells.spans[2]}, 0); //creation of tensor, therefore subsequent destruction

					move<<<LAUNCH>>> (environment, cells, future_cells); //more conversion from tensor to device_ptr
					SYNC_KERNEL(move);

					cells = future_cells; //copy assignment operator
				}
			}
		}
	}
}












