
#include"substrate_manifold.h"
#include"omega_null.h"
#include"Planar_Life.h"




namespace Cell {
	enum attribute {
		color_r = 0,
		color_g = 1,
		color_b = 2,
		attractor = 3,
		weight = 4,
		freq_a = 5,
		value = 6,
		move_maj = 7,
		move_min = 8,
	};

	const int num_attributes = 9;

}; using namespace Cell;


__global__ void draw_environment(sk::Device_Ptr<int> environment, sk::Device_Ptr<uchar> output) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());

	int environment_value = environment(maj, min);

	int polarity[3] = {0,0,0};

	if (environment_value > 0) { //positive pressure becomes red tinted
		polarity[0] += 20;
	}

	else if (environment_value < 0) { //negative pressure becomes blue tinted
		polarity[2] += 20;
	}

	int magnitude = fabsf(environment_value);

	for (int channel = 0; channel < 3; channel++) {
		output.device_data[(((channel * environment.maj()) + maj) * environment.min()) + min] = fminf(logf(10 * magnitude), 90) + polarity[channel];
	}
}


//pretty goofy way to do this to be honest. But let's see how fast or slow it runs.
__global__ void draw_cells(sk::Device_Ptr<int> cells, sk::Device_Ptr<uchar> output) {
	DIMS_2D(maj, min);
	BOUNDS_2D(cells.maj(), cells.min());

	if(cells(SELF, attribute::color_r) == 0 && cells(SELF, attribute::color_g) == 0 && cells(SELF, attribute::color_b) == 0) { return; }

	for (int channel = 0; channel < 3; channel++) {
		int channel_value = cells(maj, min, channel);

		output.device_data[(((channel * cells.maj()) + maj ) * cells.min()) + min] = channel_value;
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
	//int random_2 = curand(&local_state);
	int random_3 = curand(&local_state);
	int random_4 = curand(&local_state);
	int random_5 = curand(&local_state); 
	int random_6 = curand(&local_state);

	int attractor = Random(random_0, -10000, 10000);
	int weight = Random(random_1, -1000, 1000);
	int value = 50;
	int color_r = Random(random_3, 70, 255);
	int color_g = Random(random_4, 70, 255);
	int color_b = Random(random_5, 70, 255);
	int freq_a = Random(random_6, -10000, 10000);

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

	int weight = (cells(SELF, attribute::value) * cells(SELF, attribute::weight))/10;
	FOR_MXN_INCLUSIVE(n_maj, n_min, 9, 9, environment.maj(), environment.min(), maj, min,
		atomicAdd(&environment(n_maj, n_min), weight);
	);
}

__global__ void dampen_environment(const float damping_factor, sk::Device_Ptr<int> environment) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());

	float value = environment(maj, min);
	environment(maj, min) = truncf(value * damping_factor);


}

__global__ void radiate_environment(sk::Device_Ptr<int> environment) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());

	int value = environment(SELF);
	FOR_NEIGHBOR(n_maj, n_min, environment.maj(), environment.min(), maj, min, 
		
		atomicSub(&environment(n_maj, n_min), value/10);
	
	)
}

__global__ void set_targets(sk::Device_Ptr<int> environment, sk::Device_Ptr<int> cells, sk::Device_Ptr<int> targets) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());


	int attractor = cells(SELF, attribute::attractor);
	int weight = cells(SELF, attribute::weight);
	if(attractor == 0 && weight == 0){return;}

	int largest_value = environment(maj, min) * attractor;
	//int largest_value = -1;
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

	if(targets(maj, min) < 2) {return;}

	int teams[9][Cell::num_attributes - 2] = {0};
	int num_teams = 0;

	FOR_3X3_INCLUSIVE(n_maj, n_min, cells.maj(), cells.min(), maj, min, 
		
		if((cells(n_maj, n_min, attribute::move_maj) != maj)|| (cells(n_maj, n_min, attribute::move_min) != min)) {continue;}

		bool unassigned = true;
		for (int team = 0; team < num_teams; team++) {
			int difference = fabsf(teams[team][attribute::freq_a] - cells(n_maj, n_min, attribute::freq_a));
			if (difference < threshold) { continue; }

			unassigned = false;

			teams[team][attribute::value] += cells(n_maj, n_min, attribute::value);

			for (int attribute = 2; attribute < Cell::num_attributes - 3; attribute++) {
				teams[team][attribute] += cells(n_maj, n_min, attribute);
				teams[team][attribute] /= 2;
			}

		}
		
		for (int attribute = 0; attribute < Cell::num_attributes - 2; attribute++) {
			teams[num_teams][attribute] += cells(n_maj, n_min, attribute) * unassigned;
		}
		num_teams += unassigned
	);

	int highest_value = -1;
	int winning_team = -1;
	int total_value = 0;

	for (int i = 0; i < 9; i++) {
		total_value += teams[i][attribute::value];
		if (teams[i][attribute::value] > highest_value) {
			highest_value = teams[i][attribute::value];
			winning_team = i;
		}
	}

	future_cells(SELF, attribute::value) = total_value;
	for (int i = 0; i < Cell::num_attributes - 3; i++) {
		future_cells(SELF, i) = teams[winning_team][i];
	}
}

__global__ void move(sk::Device_Ptr<int> environment, sk::Device_Ptr<int> cells, sk::Device_Ptr<int> targets, sk::Device_Ptr<int> future_cells) {
	DIMS_2D(maj, min);
	BOUNDS_2D(environment.maj(), environment.min());

	int attractor = cells(SELF, attribute::attractor);
	int weight = cells(SELF, attribute::weight);
	int target_maj = cells(SELF, attribute::move_maj);
	int target_min = cells(SELF, attribute::move_min);

	if(targets(target_maj, target_min) != 1){return;}
	if (attractor == 0 && weight == 0) { return; }

	//#pragma unroll
	for (int i = 0; i < Cell::num_attributes; i++) {
		future_cells(target_maj, target_min, i) = cells(SELF, i);
	}

}

__global__ void hatch(const int threshold, curandState* random_states, sk::Device_Ptr<int> cells) {
	DIMS_2D(maj, min);
	BOUNDS_2D(cells.maj(), cells.min());

	int value = cells(SELF, attribute::value);
	if (value < threshold){return;} 

	int random[Cell::num_attributes];
	int id = LINEAR_CAST(maj, min, cells.min());
	curandState local_state = random_states[id];

	for (int i = 0; i < Cell::num_attributes; i++) {
		random[i] = curand(&local_state);
	}

	FOR_NEIGHBOR(n_maj, n_min, cells.maj(), cells.min(), maj, min, 
		if(cells(n_maj, n_min, attribute::attractor) == 0 && cells(n_maj, n_min, attribute::weight) == 0){
			cells(SELF, attribute::value) -= 5;

			cells(n_maj, n_min, attribute::color_r) = fmaxf( 70, fminf(cells(SELF, attribute::color_r), 255));
			cells(n_maj, n_min, attribute::color_g) = fmaxf(70, fminf(cells(SELF, attribute::color_g), 255));
			cells(n_maj, n_min, attribute::color_b) = fmaxf(70, fminf(cells(SELF, attribute::color_b), 255));
			cells(n_maj, n_min, attribute::attractor) = fmaxf(-10000, fminf(cells(SELF, attribute::attractor) + Random(random[attribute::attractor], -1, 1), 10000 ));
			cells(n_maj, n_min, attribute::weight) = fmaxf(-1000, fminf(cells(SELF, attribute::weight) + Random(random[attribute::weight], -1, 1), 1000));
			cells(n_maj, n_min, attribute::freq_a) = fmaxf(-10000, fminf(cells(SELF, attribute::freq_a) + Random(random[attribute::freq_a], -1, 1), 10000) );
			cells(n_maj, n_min, attribute::value) = 5;
			return;
		}
	);

	//cells(SELF, attribute::value) /= 2;
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

				sk::Tensor<uchar> Draw::frame(sk::Tensor<int>& cells, sk::Tensor<int>& environment) {

					sk::Tensor<uchar> output({cells.maj(), cells.min(), 3}, 0);

					sk::configure::kernel_2d(cells.maj(), cells.min());
					//draw_environment<<<LAUNCH>>>(environment, output);
					//SYNC_KERNEL(draw_environment);

					draw_cells <<<LAUNCH>>> (cells, output);
					SYNC_KERNEL(draw_cells);

					return output;

				}

				void Planar_Life::Step::polar(sk::Tensor<int>& future_cells, sk::Tensor<int>& environment, sk::Tensor<int>& cells, sk::Tensor<int>& targets, curandState* random) {
					sk::configure::kernel_2d(environment.maj(), environment.min());
					
					const int thresh = 6;
					hatch << <LAUNCH >> > (thresh, random, cells);
					SYNC_KERNEL(hatch);

					change_environment<<<LAUNCH>>> (environment, cells); 
					SYNC_KERNEL(change_environment);

					radiate_environment << <LAUNCH >> > (environment);
					SYNC_KERNEL(radiate_environment);

					const float damping_factor = 0.3;
					dampen_environment<<<LAUNCH>>>(damping_factor, environment);
					SYNC_KERNEL(dampen_environment);



					set_targets<<<LAUNCH>>>(environment, cells, targets);
					SYNC_KERNEL(set_targets);

					const int threshold = 500;
					conflict<<<LAUNCH>>>(cells, targets, future_cells, threshold);
					SYNC_KERNEL(conflict);

					move<<<LAUNCH>>> (environment, cells, targets, future_cells); 
					SYNC_KERNEL(move);



					cells = future_cells;
				}
			}
		}
	}
}












