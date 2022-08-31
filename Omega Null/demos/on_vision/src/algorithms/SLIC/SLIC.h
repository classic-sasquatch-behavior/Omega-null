#pragma once
#include"vision_manifold.h"
#include"../../vision.h"




namespace on {

	On_Structure Vision {

		On_Structure Algorithm {

			On_Process SLIC {

				static void sample_centers(Tensor<int>& source, Tensor<int>& center_pos);

				static void assign_pixels_to_centers(Tensor<int>& source, Tensor<int>& center_pos, Tensor<int>& labels);

				static void update_centers(Tensor<int>& labels, Tensor<int>& center_pos);

				static void separate_blobs(on::Tensor<int>& labels);

				static void absorb_small_blobs(on::Tensor<int>& labels);

				static void produce_ordered_labels(on::Tensor<int>& labels);

				static void enforce_connectivity(on::Tensor<int>& labels);

				static void run(Clip<int> & input, Clip<int>& output);

			};

			On_Structure Parameter {

				On_Structure SLIC {
					
					//user parameters
					const static int displacement_threshold = 1;
					const static float density = 0.5;
					const static int superpixel_size_factor = 10;

					const static int size_threshold = (superpixel_size_factor * superpixel_size_factor) / 2;

					static int source_maj, source_min, num_pixels;
					static int SP_maj, SP_min, num_superpixels;
					static int space_between_centers;
					static int density_modifier;

					static int displacement = 0;

				}
			}

		}
	}

}


