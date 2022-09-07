#pragma once
#include"vision_manifold.h"







namespace on {
	On_Structure Vision{

		template <typename Number>
		On_Being Clip {
			Clip() {}
			std::vector<sk::Tensor<Number>> frames;
		};

		On_Process Load{

			template <typename Number>
			static void clip(Clip<Number>& input, const std::string source_path) {

				fs::path resource_dir = source_path;
				for (const auto& file : fs::recursive_directory_iterator(source_path)) {

					if (file.path().extension() == ".png") {

						h_Mat new_mat = cv::imread(file.path().string(), cv::IMREAD_COLOR);
						cv::cvtColor(new_mat, new_frame, cv::COLOR_BGR2Lab);
						sk::Tensor<Number> new_frame = new_mat;
						input.frames.push_back(new_frame);

					}
				}
			}
		};

		On_Structure Window {
			
			const static std::string name = "Window";

			On_Process Display {

				template<typename Number>
				static void frame(sk::Tensor<Number>& input) {
					cv::Mat display = input;
					cv::imshow(Window::name, display);
				}

				template<typename Number>
				static void clip(Clip<Number>& input) {
					
					for (sk::Tensor<Number> frame : input.frames) {
						Display::frame(frame);
						on::Debug::wait();
					}

				}
			};
		}
	}
}