
#include"substrate_manifold.h"
#include"omega_null.h"
#include"Planar_Life.h"




namespace Cell {
	enum attribute {
		value = 0,
		attractor = 1,
		weight = 2,
		move_maj = 3,
		move_min = 4,
		color_r = 5,
		color_g = 6,
		color_b = 7,
		freq_a = 8,
	};

	const int num_attributes = 9;

}; using namespace Cell;

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

#define Random(_seed_, _min_, _max_) (((_seed_) % ((_max_) - (_min_)))+ (_min_))





__global__ void spawn(sk::Device_Ptr<int> mask, curandState* random_states, sk::Device_Ptr<int> result) {
	DIMS_2D(maj, min);
	BOUNDS_2D(result.maj(), result.min());
	//if (mask(maj, min) == 0) { return; } //mask is broken somehow, this seems to delete ALL threads rather than just half.

	int id = LINEAR_CAST(maj, min, result.min());

	curandState local_state = random_states[id];

	int random_0 = curand(&local_state);
	int random_1 = curand(&local_state);

	int attractor = Random(random_0, -1, 1);
	int weight = Random(random_1, -1, 1);
	if((attractor == 0) || (weight == 0)) {return;}

	int random_2 = curand(&local_state);
	int random_3 = curand(&local_state);
	int random_4 = curand(&local_state);
	int random_5 = curand(&local_state); 
	int random_6 = curand(&local_state);

	int value = Random(random_2, -10, 10);
	int color_r = Random(random_3, 100, 255);
	int color_g = Random(random_4, 100, 255);
	int color_b = Random(random_5, 100, 255);
	int freq_a = Random(random_6, -100, 100);

	result(SELF, attribute::value) = value;
	result(SELF, attribute::attractor) = attractor;
	result(SELF, attribute::weight) = weight;
	result(SELF, attribute::move_maj) = maj;
	result(SELF, attribute::move_min) = min;
	result(SELF, attribute::color_r) = color_r;
	result(SELF, attribute::color_g) = color_g;
	result(SELF, attribute::color_b) = color_b;
	result(SELF, attribute::freq_a) = freq_a;
}


__global__ void change_environment(sk::Device_Ptr<int> environment, sk::Device_Ptr<int> cells) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());

	int weight = cells(SELF, attribute::value) * cells(SELF, attribute::weight);
	FOR_3X3_INCLUSIVE(n_maj, n_min, environment.maj(), environment.min(), maj, min,
		atomicAdd(&environment(maj, min), weight);
	);
}

__global__ void dampen_environment(const float damping_factor, sk::Device_Ptr<int> environment) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());

	int value = environment(maj, min);
	environment(maj, min) = value * damping_factor;


}

__global__ void set_targets(sk::Device_Ptr<int> environment, sk::Device_Ptr<int> cells, sk::Device_Ptr<int> targets) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());

	int attractor = cells(SELF, attribute::attractor);

	int largest_value = environment(maj, min) * attractor;
	int target_maj = maj;
	int target_min = min;

	FOR_NEIGHBOR(n_maj, n_min, environment.maj(), environment.min(), maj, min,
		int neighbor_value = environment(n_maj, n_min) * attractor;
		if ( neighbor_value > largest_value) {
			largest_value = neighbor_value;
			target_maj = n_maj;
			target_min = n_min;
		}
	);

	atomicAdd(&targets(target_maj, target_min), 1);
	cells(SELF, attribute::move_maj) = target_maj;
	cells(SELF, attribute::move_min) = target_min;
}


__global__ void conflict(sk::Device_Ptr<int>cells, sk::Device_Ptr<int>targets, sk::Device_Ptr<int>future_cells, const int threshold = 3) {
	
	DIMS_2D(maj, min);
	BOUNDS_2D(cells.maj(), cells.min());

	int participants[9][3] = {0}; //maj, min, team
	int num_participants = 0;

	//get the cells which are participating in the conflict
	FOR_3X3_INCLUSIVE(n_maj, n_min, cells.maj(), cells.min(), maj, min, 
		int target_maj = cells(n_maj, n_min, attribute::move_maj);
		int target_min = cells(n_maj, n_min, attribute::move_min);
		if ((target_maj == maj) && target_min == min) {
			participants[num_participants][0] = n_maj;
			participants[num_participants][1] = n_min;
			num_participants++;
		}
	);

	//form teams based on freq_a
	int teams[9][2] = {0}; //freq, sum
	int num_teams = 0;
	for (int target = 0; target < num_participants; target++) {
		bool unassigned = true;
		int target_maj = participants[target][0];
		int target_min = participants[target][1];
		int target_value = cells(target_maj, target_min, attribute::value);
		int target_freq = cells(target_maj, target_min, attribute::freq_a);
		

		for (int team = 0; team < num_teams; team++) {
			if (fabsf(teams[team][0] - target_freq) < threshold) {
				unassigned = false;
				participants[target][2] = team;
				break;
			}
		}

		teams[num_teams][0] = target_freq * unassigned;
		teams[num_teams][1] += target_value * unassigned;
		num_teams += unassigned;
	}

	//the team with the highest total value wins
	int highest_value = -1;
	int winning_team = -1;

	int total_value = 0;

	for (int team = 0; team < num_teams; team++) {
		int team_value = teams[team][1];
		total_value += team_value;
		if (team_value > highest_value) {
			highest_value = team_value;
			winning_team = team;
		}
	}

	//the resulting cell becomes an average of the winners with the value of the sum of all participants
	
	int avg_attributes[Cell::num_attributes] = {0};
	int num_members = 0;

	for (int cell = 0; cell < num_participants; cell++) {
		int team = participants[cell][2];
		if(team != winning_team){continue;}

		num_members++;
		int target_maj = participants[cell][0];
		int target_min = participants[cell][1];

		#pragma unroll
		for (int attribute = 0; attribute < Cell::num_attributes; attribute++) {
			avg_attributes[attribute] = cells(target_maj, target_min, attribute);
		}

	}

	#pragma unroll
	for (int attribute = 0; attribute < Cell::num_attributes; attribute++) {
		avg_attributes[attribute] /= num_members;
	}

	avg_attributes[attribute::value] = total_value;

	#pragma unroll
	for (int attribute = 1; attribute < Cell::num_attributes; attribute++) {

		future_cells(SELF, attribute) = avg_attributes[attribute];
	}


}

__global__ void move(sk::Device_Ptr<int> environment, sk::Device_Ptr<int> cells, sk::Device_Ptr<int> targets, sk::Device_Ptr<int> future_cells) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());

	int target_maj = cells(SELF, attribute::move_maj);
	int target_min = cells(SELF, attribute::move_min);
	if(targets(target_maj, target_min) != 1){return;}

	#pragma unroll
	for (int i = 0; i < Cell::num_attributes; i++) {
		future_cells(target_maj, target_min, i) = cells(SELF, i);
	}

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

					sk::Tensor<int> result({Parameter::environment_width, Parameter::environment_height, Cell::num_attributes}, 0, "result");
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

				void Planar_Life::Step::polar(sk::Tensor<int>& future_cells, sk::Tensor<int>& environment, sk::Tensor<int>& cells, sk::Tensor<int>& targets) {
					sk::configure::kernel_2d(environment.maj(), environment.min());
					
					//const int threshold = 8;
					//hatch<<<LAUNCH>>>(threshold, cells);
					//SYNC_KERNEL(hatch);

					change_environment<<<LAUNCH>>> (environment, cells); //conversion from tensor to device_ptr
					SYNC_KERNEL(change_environment);

					//const float damping_factor = 0.99;
					//dampen_environment<<<LAUNCH>>>(damping_factor, environment);
					//SYNC_KERNEL(dampen_environment);

					

					set_targets<<<LAUNCH>>>(environment, cells, targets);
					SYNC_KERNEL(set_targets);

					const int threshold = 3;
					conflict<<<LAUNCH>>>(cells, targets, future_cells, threshold);
					SYNC_KERNEL(conflict);

					//sk::Tensor<int> future_cells({cells.spans[0], cells.spans[1], cells.spans[2]}, 0); //creation of tensor, therefore subsequent destruction

					move<<<LAUNCH>>> (environment, cells, targets, future_cells); //more conversion from tensor to device_ptr
					SYNC_KERNEL(move);

					cells = future_cells; //copy assignment operator
				}
			}
		}
	}
}












