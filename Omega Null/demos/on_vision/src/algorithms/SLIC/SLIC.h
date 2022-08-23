#pragma once
#include"vision_manifold.h"
#include"../../vision.h"




namespace on {
	On_Structure Vision{
		On_Structure Algorithm {
			On_Process SLIC {

				static void sample_centers() {

				}

				static void assign_pixels_to_centers() {

				}

				static void update_centers() {

				}

				static void separate_blobs() {

				}

				static void absorb_small_blobs() {

				}

				static void produce_ordered_labels() {

				}

				static void enforce_connectivity() {
					separate_blobs();
					absorb_small_blobs();
					produce_ordered_labels();
				}

				static void run(Clip & input, Clip & output) {
					for (Clip::Frame frame : input.frames) {
						Tensor src_L = frame.channels[0];
						Tensor src_A = frame.channels[1];
						Tensor src_B = frame.channels[2];

						Tensor flags({src_L.maj_span, src_L.min_span}, 0);
						













						output.frames.push_back(frame);
					}
				}
			};
		}
	}

}


