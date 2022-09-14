#pragma once
#include"substrate_manifold.h"
#include"omega_null.h"
#include"omega_null/display.h"






namespace on {

	On_Structure Substrate {



		On_Structure Species {

			On_Structure Planar_Life {

				On_Process Seed {
					static sk::Tensor<int> cells(int value);
				};

				On_Process Draw {
					static sk::Tensor<uchar> frame(sk::Tensor<int>& cells, sk::Tensor<int>& environment);
				};
				
				On_Structure Parameter {
					static bool running = false;
					const int environment_width = 768;
					const int environment_height = 768;
					const int environment_area = environment_width * environment_height; 
				}

				On_Process Step {
					static void polar(sk::Tensor<int>&future_cells, sk::Tensor<int>& environment, sk::Tensor<int>& cells, sk::Tensor<int>&targets, curandState* random);
				};

				static void run(sk::Tensor<int> seed = Planar_Life::Seed::cells(rand())) {

					curandState* random = on::Random::Initialize::curand_xor(Parameter::environment_area, rand());
					Planar_Life::Parameter::running = true;

					sk::Tensor<int> environment({Parameter::environment_width, Parameter::environment_height},0);
					sk::Tensor<int> cells = seed; 
					sk::Tensor<int> future_cells({Parameter::environment_width, Parameter::environment_height, 9}, 0);
					sk::Tensor<int> targets({Parameter::environment_width, Parameter::environment_height}, 0);

					sk::Tensor<uchar> frame({Parameter::environment_width, Parameter::environment_height, 3}, 0);

					//af::Window window(Parameter::environment_width, Parameter::environment_height);
					on::Display::Window::open(Parameter::environment_width, Parameter::environment_height, "Substrate");


					int start_time = now_ms();
					int FPS = 60;
					do {
						int current_time = now_ms();
						int wait_time = (1000 / FPS) - (current_time - start_time);

						Step::polar(future_cells, environment, cells, targets, random); 
						//environment.fill_device_memory(0);
						future_cells.fill_device_memory(0);
						targets.fill_device_memory(0);
						frame = Draw::frame(cells, environment); 

						on::Display::Window::render(frame);
						//window.image(frame); 
						
						std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
						start_time = now_ms();
						//std::cout << "FPS: " << 1000 / wait_time << std::endl;
					} while (Planar_Life::Parameter::running);

					on::Display::Window::close();
				}
			}
		}
	}
}






