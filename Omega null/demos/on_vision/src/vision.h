#pragma once
#include"vision_manifold.h"







namespace on {
	On_Structure Vision{

		On_Being Clip {
			Clip() {}
			std::vector<h_Mat> frames;
		};

		On_Process Load{
			static void clip(Clip& input, const std::string source_path) {

				fs::path resource_dir = source_path;
				for (const auto& file : fs::recursive_directory_iterator(source_path)) {
					if (file.path().extension() == ".png") {
						h_Mat new_frame = cv::imread(file.path().string(), cv::IMREAD_COLOR);
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

			On_Process SLIC {

				static void run(Clip& input, Clip& output) {

				}

			};

			On_Process AP{

				static void run(Clip& input, Clip& output) {

				}

			};

		}




	}
}