
#include"vision_manifold.h"
#include"omega_null.h"
#include"../../vision.h"
#include"SLIC.h"

#pragma region sample_centers

	__global__ void sample_centers_kernel(sk::Device_Ptr<int> source, sk::Device_Ptr<int> center_pos) {
		DIMS_2D(maj, min);
		BOUNDS_2D(center_pos.maj(), center_pos.min());

		center_pos(maj, min, 0) = CAST_UP(maj, center_pos.maj(), source.maj());
		center_pos(maj, min, 1) = CAST_UP(min, center_pos.min(), source.min());

	}

	__global__ void gradient_descent(sk::Device_Ptr<int> source, sk::Device_Ptr<int> center_pos) {
		DIMS_2D(maj, min);
		BOUNDS_2D(center_pos.maj(), center_pos.min());

		//gradient descent

	}

#pragma endregion

#pragma region assign_pixels_to_centers

	__global__ void pixels_to_centers(sk::Device_Ptr<int> source, sk::Device_Ptr<int> center_pos, int distance_modifier, sk::Device_Ptr<int> flags) {
		DIMS_2D(maj, min);
		BOUNDS_2D(source.maj(), source.min());

		int sector_maj = CAST_DOWN(maj, center_pos.maj());
		int sector_min = CAST_DOWN(min, center_pos.min());

		int self_channels[3] = {source(maj, min, 0), source(maj, min, 1), source(maj, min, 2)};
		int self_position[2] = {maj, min};

		int closest_center = -1;
		int smallest_distance = INT_MAX;

		FOR_3X3_INCLUSIVE(n_maj, n_min, center_pos.maj(), center_pos.min(), maj, min, 
			int center_id = LINEAR_CAST(n_maj, n_min, center_pos.min());
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
	
	__global__ void tally_centers(sk::Device_Ptr<int> flags, sk::Device_Ptr<int> tally) {
		DIMS_2D(maj, min);
		BOUNDS_2D(flags.maj(), flags.min());

		int id = flags(maj, min);

		atomicAdd(&tally(id, 0), maj);
		atomicAdd(&tally(id, 1), min);
		atomicAdd(&tally(id, 2), 1);

	}

	__global__ void move_centers(sk::Device_Ptr<int> tally, sk::Device_Ptr<int> center_pos, int* displacement) {
		DIMS_2D(id, ZERO);
		BOUNDS_2D(tally.maj(), 1);
		
		int min = id % center_pos.min();
		int maj = (id - min) / center_pos.min();

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

	__global__ void separate_blobs_kernel(sk::Device_Ptr<int> labels, sk::Device_Ptr<int> flag, sk::Device_Ptr<int> blobs) {
		DIMS_2D(maj, min);
		BOUNDS_2D(labels.maj(), labels.min());

		int id = LINEAR_CAST(maj, min, labels.min());
		int label = labels(maj, min);
		int blob = blobs(maj, min);

		FOR_NEIGHBOR(n_maj, n_min, labels.maj(), labels.min(), maj, min,		
			int neighbor_label = labels(maj, min);
			int neighbor_blob = blobs(maj, min);
			if (neighbor_label != label) {continue;}
			
			if (neighbor_blob > blob) { 
				blob = neighbor_blob;
			}
		);

		if (blob != blobs(maj, min)) {
			blobs(maj, min) = blob;
			flag(0) = 1; //devise a way to assign a flag from the device without parentheses
		}
	}

#pragma endregion

#pragma region absorb_small_blobs
	
	__global__ void find_sizes() {

	}

	__global__ void find_weak_labels() {

	}

	__global__ void absorb_small_blobs_kernel() {

	}

#pragma endregion

#pragma region produce_ordered_labels

	__global__ void raise_flags() {

	}

	__global__ void init_map() {

	}

	__global__ void invert_map() {

	}

	__global__ void assign_new_labels() {

	}

#pragma endregion

namespace on {

	On_Structure Vision {

		using namespace Algorithm::Parameter::SLIC;
		On_Structure Algorithm {
			
			void SLIC::sample_centers(sk::Tensor<int>& source, sk::Tensor<int>& center_pos) {

				sk::configure::kernel_2d(center_pos.maj(), center_pos.min());
				sample_centers_kernel<<<LAUNCH>>>(source, center_pos);
				SYNC_KERNEL(sample_centers);

				sk::configure::kernel_2d(center_pos.maj(), center_pos.min());
				gradient_descent<<<LAUNCH>>>(source, center_pos);
				SYNC_KERNEL(gradient_descent);

			}

			void SLIC::assign_pixels_to_centers(sk::Tensor<int>& source, sk::Tensor<int>& center_pos, sk::Tensor<int>& labels) {

				sk::configure::kernel_2d(source.maj(), source.min());
				pixels_to_centers<<<LAUNCH>>>(source, center_pos, density_modifier, labels);
				SYNC_KERNEL(pixels_to_centers);

			}

			void SLIC::update_centers(sk::Tensor<int>& labels, sk::Tensor<int>& center_pos) {

				//0 = maj sum , 1 = min sum, 2 = count
				sk::Tensor<int> tally({(uint)Parameter::SLIC::num_superpixels, 3});

				sk::configure::kernel_2d(labels.maj(), labels.min());
				tally_centers<<<LAUNCH>>>(labels, tally);
				SYNC_KERNEL(tally_centers);

				sk::Tensor<int> temp_displacement;
				sk::configure::kernel_1d(tally.maj());
				move_centers<<<LAUNCH>>>(tally, center_pos, temp_displacement); 
				SYNC_KERNEL(move_centers);
				Parameter::SLIC::displacement = temp_displacement;
					
			}

			#pragma region enforce connectivity
			void SLIC::separate_blobs(sk::Tensor<int>& labels) {

				sk::Tensor<int> flag;
				sk::Tensor<int> blobs({labels.maj(), labels.min()}, 0);
				blobs = labels;

				sk::configure::kernel_2d(labels.maj(), labels.min());

				do {
					separate_blobs_kernel<<<LAUNCH>>>(labels, flag, blobs);
					SYNC_KERNEL(separate_blobs);

				} while (flag == 1);

				labels = blobs;
			}
				
			void SLIC::absorb_small_blobs(sk::Tensor<int>& labels) {
						
				find_sizes<<<LAUNCH>>>();
				SYNC_KERNEL(find_sizes);

				find_weak_labels<<<LAUNCH>>>();
				SYNC_KERNEL(find_weak_labels);

				do {

					absorb_small_blobs_kernel<<<LAUNCH>>>();
					SYNC_KERNEL(absorb_small_blobs);

				} while (0);

			}

			void SLIC::produce_ordered_labels(sk::Tensor<int>& labels) {

				raise_flags<<<LAUNCH>>>();
				SYNC_KERNEL(raise_flags);

				init_map<<<LAUNCH>>>();
				SYNC_KERNEL(init_map);

				invert_map<<<LAUNCH>>>();
				SYNC_KERNEL(invert_map);

				assign_new_labels<<<LAUNCH>>>();
				SYNC_KERNEL(assign_new_labels);

			}
			#pragma endregion

			void SLIC::enforce_connectivity(sk::Tensor<int>& labels) {
				separate_blobs(labels);
				absorb_small_blobs(labels);
				produce_ordered_labels(labels);
			}

			void SLIC::run(Clip<int>& input, Clip<int>& output) {
				for (sk::Tensor source : input.frames) {

					source_maj = source.maj();
					source_min = source.min();
					num_pixels = source_maj * source_min;

					SP_maj;
					SP_min;
					num_superpixels;

					space_between_centers;
					density_modifier;

					sk::Tensor<int> labels({ (uint)source_maj, (uint)source_min }, 0);

					sk::Tensor<int> center_pos({(uint)SP_maj, (uint)SP_min, (uint)2}, 0); //z = 0 is maj, z = 1 is min

					sample_centers(center_pos, source);

					do {

						assign_pixels_to_centers(source, center_pos, labels);

						update_centers(labels, center_pos);

					} while (displacement < displacement_threshold);

					enforce_connectivity(labels);

					output.frames.push_back(labels);
				}
				return;
			}
		}
	}

}