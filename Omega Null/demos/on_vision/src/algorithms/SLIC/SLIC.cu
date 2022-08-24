
#include"vision_manifold.h"
#include"../../vision.h"
#include"SLIC.h"


#pragma region sample_centers

	__global__ kernel sample_centers() {
		
	}

	__global__ kernel gradient_descent() {

	}

#pragma endregion

#pragma region assign_pixels_to_centers


#pragma endregion

#pragma region update_centers


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

			On_Process SLIC {

				void SLIC::sample_centers(Tensor<int>& center_maj, Tensor<int>& center_min) {

					on::Launch::Kernel::conf_2d(center_maj.maj_span, center_maj.min_span);
					sample_centers<<<LAUNCH>>>();
					On_Sync(sample_centers);

					on::Launch::Kernel::conf_2d(center_maj.maj_span, center_maj.min_span);
					gradient_descent<<<LAUNCH>>>();
					On_Sync(gradient_descent);

				}

				void SLIC::assign_pixels_to_centers() {

				}

				void SLIC::update_centers() {

				}

				void SLIC::separate_blobs() {

				}

				void SLIC::absorb_small_blobs() {

				}

				void SLIC::produce_ordered_labels() {

				}

				void SLIC::enforce_connectivity() {
					separate_blobs();
					absorb_small_blobs();
					produce_ordered_labels();
				}

				void SLIC::run(Clip<int>& input, Clip<int>& output) {
					for (Clip<int>::Frame frame : input.frames) {
						Clip<int>::Frame processed_frame;
						Tensor<int> src_L = frame.channels[0];
						Tensor<int> src_A = frame.channels[1];
						Tensor<int> src_B = frame.channels[2];

						Parameter::SLIC::source_maj;
						Parameter::SLIC::source_min;
						Parameter::SLIC::num_pixels;

						Parameter::SLIC::SP_maj;
						Parameter::SLIC::SP_min;
						Parameter::SLIC::num_superpixels;

						Parameter::SLIC::space_between_centers;
						Parameter::SLIC::density_modifier;

						Tensor<int> flags({ src_L.maj_span, src_L.min_span }, 0);

						//what is this structure here? essentially, 2 channel matrix. maybe we can set up a way to support channels in tensor? -> Tensor(x,y)[channel] - ezpz
						//each channel would be actually stored in its own array. so you would have one host array and one device array for each channel. actually not quite. 
						//you would need to store n * 2 arrays, but synchronization would be controlled by exactly two states/objects. i.e. the n combined channels would be 
						//treated as one unit on the host/device for the purposes of synchronization. 
						Tensor<int> center_maj({(uint)Parameter::SLIC::SP_maj, (uint)Parameter::SLIC::SP_min}, 0);
						Tensor<int> center_min({(uint)Parameter::SLIC::SP_maj, (uint)Parameter::SLIC::SP_min }, 0);


						sample_centers(center_maj, center_min);

						do {

							assign_pixels_to_centers();

							update_centers();

						} while (Parameter::SLIC::displacement < Parameter::SLIC::displacement_threshold);

						enforce_connectivity();

						processed_frame.channels.push_back(flags);
						output.frames.push_back(processed_frame);
					}
					return;
				}
			};
		}
	}

}