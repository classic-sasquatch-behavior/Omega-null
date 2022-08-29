#pragma once

#include"substrate_manifold.h"








namespace on {

	On_Structure Substrate {

		enum Display_Flag {

		};

		template<typename Type>
		On_Being Frame {
			std::vector<on::Tensor<Type>> planes;
			std::vector<Display_Flag> flags;
		};

		On_Process Draw{
			template<typename Type>
			static Frame<Type> frame(std::vector<on::Tensor<Type>> input) {
				Frame<Type> result;

				return result;
			}
		};

		On_Process Display {
			template <typename Type>
			static void frame(Frame<Type>& input) {
				for (int i = 0; i < input.size; i++) {
					on::Tensor<Type> plane = input.planes[i]
					Display_Flag flag = input.flags[i]

					switch (flag) {
						case 0: break;
						default: break;
					}

					//render to window



				}
			}
		};

		On_Process Listen{

			static void for_input() {

			}

		}


	}

}








