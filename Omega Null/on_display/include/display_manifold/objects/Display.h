#pragma once

#include"display_manifold.h"




namespace on {
	
	On_Structure Display {

		On_Structure Window {
			
			On_Structure View { //to contain the information for OpenGL

			}

			inline GLFWwindow* window;

			static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
				std::cout << std::endl << "scroll detected: " << yoffset << std::endl;
			}

			static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
				switch (key) {
					case GLFW_KEY_W: std::cout << "w" << std::endl; break;
					case GLFW_KEY_A: std::cout << "a" << std::endl; break;
					case GLFW_KEY_S: std::cout << "s" << std::endl; break;
					case GLFW_KEY_D: std::cout << "d" << std::endl; break;
					default: std::cout << "unknown key" << std::endl; break;
				}
			}

			static void open(const uint width, const uint height, std::string title) {
				glfwInit();
				window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);

				glfwSetKeyCallback(window, key_callback);
				glfwSetScrollCallback(window, scroll_callback);

				glfwMakeContextCurrent(window);
			}

			static void render(sk::Tensor<uchar> input) {

				//glClear(GL_COLOR_BUFFER_BIT);
				
				//opengl rendering goes here
				//want to do the whole textured quad thing. want to store viewport position and render the image in accordance with it.

				glfwSwapBuffers(window);
				glfwPollEvents();
			}

			static void close() {
				glfwDestroyWindow(window);
				glfwTerminate();
			}	

		}


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

			static void render(sk::Tensor<uchar>& input) {
				
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

			static void render(std::string window_name, sk::Tensor<uchar>& input) {

			}
		}
	}
}