#pragma once

#include"display_manifold.h"




namespace on {
	
	On_Structure Display {

		On_Structure Forge {

			//inline af::Window window;

			On_Process Initialize {
				static void window(int width, int height) {
					
				}
			};

			On_Process Listen {

				static void for_input() {
					
					

				}
			};

			static void render(on::Tensor<uchar>& input) {
				
				af::array frame = input;
			
				//Forge::window.image(input);

			}
		}

		On_Structure OpenCV {

			On_Process Initialize {

				static void window() {

				}

			};

			On_Process Listen {
				
				static void for_input() {

				}
			};

			static void render(std::string window_name, on::Tensor<uchar>& input) {

			}
		}
	}
}