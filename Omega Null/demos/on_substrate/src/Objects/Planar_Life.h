#pragma once
#include"substrate_manifold.h"
#include"omega_null.h"
#include"omega_null/display.h"






namespace on {

	On_Structure Substrate {

		On_Structure Species {

			On_Structure Planar_Life {

				On_Process Seed {
					static on::Tensor<int> cells(int value);
				};

				On_Process Draw {
					static on::Tensor<uchar> frame(on::Tensor<int>& cells);
				};
				
				On_Structure Parameter {
					static bool running = false;
					const int environment_width = 512;
					const int environment_height = 512;
					const int environment_area = environment_width * environment_height; 
				}

				On_Process Step {
					static void polar(Tensor<int>& environment, Tensor<int>& cells);
				};

				static void run(on::Tensor<int> seed = Planar_Life::Seed::cells(rand())) {

					Planar_Life::Parameter::running = true;

					on::Tensor<int> environment({Parameter::environment_width, Parameter::environment_height},0);
					on::Tensor<int> cells = seed; //channel 0: values //channel 1: attractors

					on::Tensor<uchar> frame({Parameter::environment_width, Parameter::environment_height, 3}, 0);

					af::Window window(Parameter::environment_width, Parameter::environment_height);

					int start_time = now_ms();
					int FPS = 20;
					do {
						int current_time = now_ms();
						int wait_time = (1000 / FPS) - (current_time - start_time);



						Step::polar(environment, cells); 
						//conversion from tensor to device_ptr
						//creation and destruction of tensor
						//tensor copy assignment operator

						frame = Draw::frame(cells); 
						//conversion from tensor to device_ptr
						//creation and destruction of tensor 
						//tensor copy assignment operator


						window.image(frame); //conversion from tensor to af::array 

						std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
						start_time = now_ms();
						//std::cout << "FPS: " << 1000 / wait_time << std::endl;
					} while (Planar_Life::Parameter::running);

				}
			}
		}
	}
}






