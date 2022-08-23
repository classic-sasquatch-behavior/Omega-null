#pragma once
#include"vision_manifold.h"







namespace on {
	On_Structure Vision{

		On_Being Clip {

			Clip() {}

				On_Being Frame {

					Frame() {}

					std::vector<Tensor<uchar>> channels;

				};

			std::vector<Frame> frames;

		};

		On_Process Load{
			static void clip(Clip& input, const std::string source_path) {

				fs::path resource_dir = source_path;
				for (const auto& file : fs::recursive_directory_iterator(source_path)) {
					if (file.path().extension() == ".png") {
						h_Mat new_frame_bgr = cv::imread(file.path().string(), cv::IMREAD_COLOR);
						//convert frame to LAB*
						cv::cvtColor(new_frame_bgr, new_frame_bgr, cv::COLOR_BGR2Lab);
						std::vector<h_Mat> split_frame_lab;
						cv::split(new_frame_bgr,split_frame_lab);
						
						Clip::Frame new_frame;

						for (int i = 0; i < split_frame_lab.size(); i++) {
							on::Tensor<uchar> channel;
							
							channel = split_frame_lab[i];

							new_frame.channels.push_back(channel);
						}

						input.frames.push_back(new_frame);

					}
				}
			}
		};

		On_Process Display{
			static void clip(Clip & input) {

			}
		};

		On_Structure Algorithm {




		}




	}
}