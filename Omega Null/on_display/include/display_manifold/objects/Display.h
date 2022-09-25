#pragma once

#include"display_manifold.h"


//static const char* vertex_shader_source = "#version 330 core\n"
//"layout (location = 0) in vec3 aPos;\n"
//"void main()\n"
//"{\n"
//"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
//"}\0";
//
//static const char* fragment_shader_source = "#version 330 core\n"
//"out vec4 FragColor;\n"
//"void main()\n"
//"{\n"
//"	FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
//"}\0";

static const char* texture_vertex_shader = "#version 330 core\n"
"layout(location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec3 aColor;\n"
"layout(location = 2) in vec2 aTexCoord;\n"
"out vec3 ourColor;\n"
"out vec2 TexCoord;\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(aPos, 1.0);\n"
"	ourColor = aColor;\n"
"	TexCoord = aTexCoord;\n"
"}\0";

static const char* texture_fragment_shader = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 ourColor;\n"
"in vec2 TexCoord;\n"
"uniform sampler2D ourTexture;\n"
"void main()\n"
"{\n"
"	FragColor = texture(ourTexture, TexCoord);\n"
"}\0";

static const char* texture_vertex_shader_zoom = "#version 330 core\n"
"layout(location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec2 aTexCoord;\n"
"out vec2 TexCoord;\n"
"uniform mat4 transform;\n"
"void main()\n"
"{\n"
"gl_Position = transform * vec4(aPos, 1.0f);\n"
"TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
"} \0";



namespace on {
	
	On_Structure Display {

		On_Structure Window {
			
			On_Structure Surface { //to contain the information for OpenGL

				On_Structure Object {
					static gl_name texture;
						
					static gl_name vertex_buffer_object;
					static gl_name vertex_array_object;

					static gl_name vertex_shader;
					static gl_name fragment_shader;
					static gl_name shader_program;

					static gl_name element_buffer_object;

					//static float triangle[] {
					//	-0.5f, -0.5f, 0.0f,
					//	0.5f, -0.5f, 0.0f,
					//	0.0f, 0.5f, 0.0f
					//};

					//static float square[] = {
					//	1.0f,  1.0f, 0.0f,  // top right
					//	1.0f, -1.0f, 0.0f,  // bottom right
					//	-1.0f, -1.0f, 0.0f,  // bottom left
					//	-1.0f,  1.0f, 0.0f   // top left 
					//};
					static uint square_indices[] = {  // note that we start from 0!
						0, 1, 3,   // first triangle
						1, 2, 3    // second triangle
					};

					static float texture_vertices[] = {
						1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // top right
						1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
					   -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,   // bottom left
					   -1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f    // top left 
					};
				}

				On_Process Initialize {

					static void texture() {
						glGenBuffers(1, &Object::vertex_buffer_object);
						glBindBuffer(GL_ARRAY_BUFFER, Object::vertex_buffer_object);
						glBufferData(GL_ARRAY_BUFFER, sizeof(Object::texture_vertices), Object::texture_vertices, GL_STATIC_DRAW);
						glGenTextures(1, &Object::texture);
						glBindTexture(GL_TEXTURE_2D, Object::texture);

					}

					//static void triangle() {
					//	glGenBuffers(1, &Object::vertex_buffer_object);
					//	glBindBuffer(GL_ARRAY_BUFFER, Object::vertex_buffer_object);
					//	glBufferData(GL_ARRAY_BUFFER, sizeof(Object::triangle), Object::triangle, GL_STATIC_DRAW);
					//}

					//static void square() {
					//	glGenBuffers(1, &Object::vertex_buffer_object);
					//	glBindBuffer(GL_ARRAY_BUFFER, Object::vertex_buffer_object);
					//	glBufferData(GL_ARRAY_BUFFER, sizeof(Object::square), Object::square, GL_STATIC_DRAW);
					//}

					//static void triangle_vao() {
					//	glGenVertexArrays(1, &Object::vertex_array_object);
					//	glBindVertexArray(Object::vertex_array_object);
					//	glBindBuffer(GL_ARRAY_BUFFER, Object::vertex_buffer_object);
					//	glBufferData(GL_ARRAY_BUFFER, sizeof(Object::triangle), Object::triangle, GL_STATIC_DRAW);
					//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
					//	glEnableVertexAttribArray(0);
					//}

					//static void square_vao() {
					//	glGenVertexArrays(1, &Surface::Object::vertex_array_object);
					//	glBindVertexArray(Surface::Object::vertex_array_object);
					//}

					static void vertex_attribute_pointers() {
						glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
						glEnableVertexAttribArray(0);
					}

					static void vertex_shader(const char* source) {
						Object::vertex_shader = glCreateShader(GL_VERTEX_SHADER);
						glShaderSource(Object::vertex_shader, 1, &source, NULL);
						glCompileShader(Object::vertex_shader);
					}
					
					static void fragment_shader(const char* source) {
						Object::fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
						glShaderSource(Object::fragment_shader, 1, &source, NULL);
						glCompileShader(Object::fragment_shader);
					}

					static void shader_program() {
						Object::shader_program = glCreateProgram();
						glAttachShader(Object::shader_program, Object::vertex_shader);
						glAttachShader(Object::shader_program, Object::fragment_shader);
						glLinkProgram(Object::shader_program);
						glUseProgram(Object::shader_program);
						glDeleteShader(Object::vertex_shader);
						glDeleteShader(Object::fragment_shader);
					}

					//static void element_buffer_object() {
					//	glGenBuffers(1, &Surface::Object::element_buffer_object);
					//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Surface::Object::element_buffer_object);
					//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Surface::Object::square_indices), Surface::Object::square_indices, GL_STATIC_DRAW);

					//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
					//	glEnableVertexAttribArray(0);
					//}

					static void texture_ebo() {
						glGenBuffers(1, &Surface::Object::element_buffer_object);
						glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Surface::Object::element_buffer_object);
						glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Surface::Object::square_indices), Surface::Object::square_indices, GL_STATIC_DRAW);

						glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(0 * sizeof(float)));
						glEnableVertexAttribArray(0);
						glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
						glEnableVertexAttribArray(1);
						glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
						glEnableVertexAttribArray(2);

					}

				};

				On_Process Draw{
					static void triangle() {
						glDrawArrays(GL_TRIANGLES, 0, 3);
					}

					static void square() {
						glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
						glBindVertexArray(0);
					}
				};

			}

			inline GLFWwindow* window;
			//static gl_name glew;

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
				glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

				glfwSetKeyCallback(window, key_callback);
				glfwSetScrollCallback(window, scroll_callback);

				glfwMakeContextCurrent(window);
				glewInit();

				Surface::Initialize::vertex_shader(texture_vertex_shader);
				Surface::Initialize::fragment_shader(texture_fragment_shader);
				Surface::Initialize::shader_program();

				//Surface::Initialize::triangle();
				//Surface::Initialize::triangle_vao();

				//Surface::Initialize::square();
				//Surface::Initialize::square_vao();
				//Surface::Initialize::element_buffer_object();

				Surface::Initialize::texture();
				Surface::Initialize::texture_ebo();

				//glfwSwapInterval(0);

			}

			static void render(sk::Tensor<uchar>& input) {

				uchar* data = input.data(sk::host);

				//glm::mat4 transform = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
				//transform = glm::translate(transform, glm::vec3(0.5f, -0.5f, 0.0f));
				//unsigned int transformLoc = glGetUniformLocation(Surface::Object::shader_program, "transform");
				//glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(transform));

				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, input.first_dim(), input.second_dim(), 0, GL_RGB, GL_UNSIGNED_BYTE, data);
				
				glGenerateMipmap(GL_TEXTURE_2D);

				glClear(GL_COLOR_BUFFER_BIT);

				glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

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