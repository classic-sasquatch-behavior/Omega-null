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
						//std::cout << file.path().string() << std::endl;
						h_Mat new_frame = cv::imread(file.path().string(), cv::IMREAD_COLOR);
						input.frames.push_back(new_frame);
					}
				}
			}
		};






	}
}