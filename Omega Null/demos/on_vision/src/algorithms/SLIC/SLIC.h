#pragma once
#include"vision_manifold.h"
#include"../../vision.h"




namespace on {

	On_Structure Vision {

		On_Structure Algorithm {

			On_Process SLIC {

				static void sample_centers();

				static void assign_pixels_to_centers();

				static void update_centers();

				static void separate_blobs();

				static void absorb_small_blobs();

				static void produce_ordered_labels();

				static void enforce_connectivity();

				static void run(Clip<int> & input, Clip<int>& output);

			};

			On_Structure Parameter {

				On_Structure SLIC {
					

					static int displacement_threshold = 1;
					static float density = 0.5;
					static int superpixel_size_factor = 10;
					static int size_threshold = (superpixel_size_factor * superpixel_size_factor) / 2;

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


