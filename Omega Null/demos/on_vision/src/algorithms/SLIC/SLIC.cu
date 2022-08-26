
#include"vision_manifold.h"
#include"../../vision.h"
#include"SLIC.h"


#pragma region sample_centers

	__global__ void sample_centers(on::Tensor<int> source, on::Tensor<int> center_pos) {
		GET_DIMS(maj, min);
		CHECK_BOUNDS(center_pos.maj_span, center_pos.min_span);

		center_pos(maj, min, 0) = CAST_UP(maj, center_pos.maj_span, source.maj_span);
		center_pos(maj, min, 1) = CAST_UP(min, center_pos.min_span, source.min_span);

	}

	__global__ void gradient_descent(on::Tensor<int> source, on::Tensor<int> center_pos) {
		GET_DIMS(maj, min);
		CHECK_BOUNDS(center_pos.maj_span, center_pos.min_span);

		//gradient descent

	}

#pragma endregion

#pragma region assign_pixels_to_centers

	__global__ void pixels_to_centers(on::Tensor<int> source, on::Tensor<int> center_pos, int distance_modifier, on::Tensor<int> flags) {
		GET_DIMS(maj, min);
		CHECK_BOUNDS(source.maj_span, source.min_span);

		int sector_maj = CAST_DOWN(maj, center_pos.maj_span);
		int sector_min = CAST_DOWN(min, center_pos.min_span);

		int self_channels[3] = {source(maj, min, 0), source(maj, min, 1), source(maj, min, 2)};
		int self_position[2] = {maj, min};

		int closest_center = -1;
		int smallest_distance = INT_MAX;

		FOR_3X3_INCLUSIVE(n_maj, n_min, center_pos.maj_span, center_pos.min_span, maj, min, 
			int center_id = LINEAR_CAST(n_maj, n_min, center_pos.min_span);
			int neighbor_maj = center_pos(n_maj, n_min, 0);
			int neighbor_min = center_pos(n_maj, n_min, 1);

			int neighbor_channels[3] = {source(neighbor_maj, neighbor_min, 0), source(neighbor_maj, neighbor_min, 1), source(neighbor_maj, neighbor_min, 2)};
			int neighbor_position[2] = {neighbor_maj, neighbor_min};

			int color_distance = 0;
			for (int channel = 0; channel < 3; channel++) {
				color_distance += (neighbor_channels[channel] - self_channels[channel]) * (neighbor_channels[channel] - self_channels[channel]);
			}
		
			int spatial_distance = 0;
			for (int coordinate = 0; coordinate < 2; coordinate++) {
				spatial_distance += (neighbor_position[coordinate] - neighbor_position[coordinate]) * (neighbor_position[coordinate] - neighbor_position[coordinate]);
			}

			int neighbor_distance = color_distance + (distance_modifier * spatial_distance);

			if (neighbor_distance < smallest_distance) {
				closest_center = center_id;
				smallest_distance = neighbor_distance;
			}
		);

		int result = closest_center;
		flags(maj, min) = result; 
		return;
	}

#pragma endregion

#pragma region update_centers
	
	__global__ void tally_centers(on::Tensor<int> flags, on::Tensor<int> tally) {
		GET_DIMS(maj, min);
		CHECK_BOUNDS(flags.maj_span, flags.min_span);

		int id = flags(maj, min);

		atomicAdd(&tally(id, 0), maj);
		atomicAdd(&tally(id, 1), min);
		atomicAdd(&tally(id, 2), 1);

	}

	__global__ void move_centers(on::Tensor<int> tally, on::Tensor<int> center_pos, int* displacement) {
		GET_DIMS(id, ZERO);
		CHECK_BOUNDS(tally.maj_span, 1);
		
		int maj = id % center_pos.min_span;
		int min = (id - min) / center_pos.min_span;

		int old_maj = center_pos(maj, min, 0);
		int old_min = center_pos(maj, min, 1);

		int maj_sum = tally(id, 0);
		int min_sum = tally(id, 1);
		int size = tally(id, 2);

		int new_maj = maj_sum / size;
		int new_min = min_sum / size;

		int this_displacement = sqrtf(((old_maj - new_maj) * (old_maj - new_maj)) + ((old_min - new_min) * (old_min - new_min)));

		atomicAdd(displacement, this_displacement);

		center_pos(min, maj, 0) = new_maj;
		center_pos(min, maj, 1) = new_min;
		
	}

#pragma endregion

#pragma region separate_blobs


#pragma endregion

#pragma region absorb_small_blobs


#pragma endregion

#pragma region produce_ordered_labels


#pragma endregion

namespace on {

	On_Structure Vision{

		On_Structure Algorithm {
			
			using namespace Parameter::SLIC;
			On_Process SLIC {
				

				void SLIC::sample_centers(Tensor<int>& source, Tensor<int>& center_pos) {

					Launch::Kernel::conf_2d(center_pos.maj_span, center_pos.min_span);
					sample_centers<<<LAUNCH>>>(source, center_pos);
					On_Sync(sample_centers);

					Launch::Kernel::conf_2d(center_pos.maj_span, center_pos.min_span);
					gradient_descent<<<LAUNCH>>>(source, center_pos);
					On_Sync(gradient_descent);

				}

				void SLIC::assign_pixels_to_centers(Tensor<int>& source, Tensor<int>& center_pos, Tensor<int>& flags) {

					Launch::Kernel::conf_2d(source.maj_span, source.min_span);
					pixels_to_centers<<<LAUNCH>>>(source, center_pos, , flags);
					On_Sync(pixels_to_centers);

				}

				void SLIC::update_centers(Tensor<int>& flags, Tensor<int>& center_pos) {

					//0 = maj sum , 1 = min sum, 2 = count
					on::Tensor<int> tally({(uint)Parameter::SLIC::num_superpixels, 3});

					Launch::Kernel::conf_2d(flags.maj_span, flags.min_span);
					tally_centers<<<LAUNCH>>>(flags, tally);
					On_Sync(tally_centers);

					int* temp_d_displacement;
					cudaMalloc(&temp_d_displacement, sizeof(int));

					set_flag(temp_displacement);

					Launch::Kernel::conf_1d(tally.maj_span);
					move_centers<<<LAUNCH>>>(tally, center_pos, temp_d_displacement); 
					On_Sync(update_centers);

					get_flag(temp_displacement);

					int temp_h_displacement;
					cudaMemcpy(&temp_h_displacement, temp_d_displacement, sizeof(int), cudaMemcpyDeviceToHost);
					cudaFree(temp_d_displacement);

					Parameter::SLIC::displacement = temp_h_displacement;
					
				}

				#pragma region enforce connectivity
					void SLIC::separate_blobs() {

					}

					void SLIC::absorb_small_blobs() {

					}

					void SLIC::produce_ordered_labels() {

					}
				#pragma endregion

				void SLIC::enforce_connectivity(on::Tensor<int>& flags) {
					separate_blobs();
					absorb_small_blobs();
					produce_ordered_labels();
				}

				void SLIC::run(Clip<int>& input, Clip<int>& output) {
					for (Tensor source : input.frames) {

						source_maj = source.maj_span;
						source_min = source.min_span;
						num_pixels = source_maj * source_min;

						SP_maj;
						SP_min;
						num_superpixels;

						space_between_centers;
						density_modifier;

						Tensor<int> flags({ (uint)source_maj, (uint)source_min }, 0);

						//z = 0 is maj, z = 1 is min
						Tensor<int> center_pos({(uint)SP_maj, (uint)SP_min, (uint)2}, 0);


						sample_centers(center_pos, source);

						do {

							assign_pixels_to_centers(source, center_pos, flags);

							update_centers(flags, center_pos);

						} while (Parameter::SLIC::displacement < Parameter::SLIC::displacement_threshold);

						enforce_connectivity(flags);

						output.frames.push_back(flags);
					}
					return;
				}
			};
		}
	}

}