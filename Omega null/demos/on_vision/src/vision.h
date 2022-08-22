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
				for (auto& file : std::filesystem::recursive_directory_iterator(resource_dir)) {
					if (file.path().extension() == ".png") {
						h_Mat new_frame = cv::imread(file.path().string());
						input.frames.push_back(new_frame);
					}
				}
			}
		};






	}
}