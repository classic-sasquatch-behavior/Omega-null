#pragma once
#include"global_manifold.h"
#include"omega_null.h"

namespace on {
	On_Structure Debug {
		enum strangeness {
			likely = 0,
			plausible = 1,
			unknown = 2,
			unlikely = 3,
			impossible = 4
		};

		static cudaError_t cuda_error;

		On_Being Error {
			Debug::strangeness strangeness;
			std::string comment;
			std::string file;
			uint line;
			Error() : comment(""), strangeness(unknown), file("unspecified"), line(0) {}
		};

		On_Process Throw {
			static void error(Debug::Error & error) {
				std::string message = "ON::ERROR : ";
				message += error.comment;
				message += " : this happened in file : ";
				message += error.file;
				message += " : at line : ";
				message += error.line;

				switch (error.strangeness) {
					case likely: message += " : you knew that would happen [0/4]"; break;
					case plausible: message += " : you thought that might happen [1/4]"; break;
					case unknown: message += " : you weren't sure if that would happen [2/4]"; break;
					case unlikely: message += " : you didn't think that would happen [3/4]"; break;
					case impossible: message += " : that definitely wasn't supposed to happen [4/4]"; break;
				default: message += " : ...what? [?/ 4]"; break;
				}
				std::cout << "\n" + message + "\n";
			}
		};

		
		On_Process Print{
			template <typename ElementType>
			static void tensor(on::Tensor<ElementType> &input, uint max_depth = 10) {
				std::cout << std::endl << input.num_dims << " dimensional matrix:" << std::endl;
				
				for (int i = 0; (i < max_depth) && (i < input.spans[0]); i++) {
					std::cout << std::endl;
					for (int j = 0; (j < max_depth) && (j < input.spans[1]); j++) {
						std::cout << input(i, j) << ", ";
					}
					if (input.spans[1] > max_depth) {
						std::cout << "..." << std::endl;
					}
				}
				std::cout << std::endl;
			}

			template <typename ElementType>
			static void vector(std::vector<ElementType>& input, uint max_depth = 10) {
				std::cout << std::endl << input.size() << " element long vector: ";
				
				for (int i = 0; (i < max_depth) && (i < input.size()); i++) {
					std::cout << input[i] << ", ";
				}

				if (input.size() > max_depth) {
					std::cout << "...";
				}
				
				std::cout << std::endl;
			}

			static void af_array(const char* message, af::array& input){
				af::print(message, input);
			}

			template <typename ElementType>
			static void h_Mat(cv::Mat& input, uint max_depth = 10) {
				std::cout << std::endl << "printing cv::Mat" << std::endl;

				int num_channels = input.channels();
				
				std::cout << std::endl;

				for (int row = 0; (row < input.rows) && (row < max_depth); row++) {
					for (int col = 0; (col < input.cols) && (col < max_depth); col++) {

						ElementType element = input.at<ElementType>(row, col);

						std::cout << element << ", ";


						//for (int channel = 0; channel < num_channels; channel++) {
						//}
					}
					if (input.cols > max_depth) {
						std::cout << "...";
					}
					std::cout << std::endl;
					
				}




			}

			template <typename ElementType>
			static void d_Mat(cv::cuda::GpuMat& input) {
				std::cout << std::endl << "printing device mat";
				cv::Mat temp;
				input.download(temp);
				Print::h_Mat<ElementType>(temp);
			}

		};

		static void wait() {
			std::cin.get();
		}

	}
}





