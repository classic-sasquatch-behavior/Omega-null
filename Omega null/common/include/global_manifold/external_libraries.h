#pragma once


//omega null


//filestreams
#include<iostream>
#include<fstream>
#include<sstream>
typedef std::ofstream Filestream;

//standard lib
#include<queue>
#include<string>

//filesystem
#include<filesystem>
namespace fs = std::filesystem;


//opengl

//glew
#include<GL/glew.h>
typedef unsigned int gl_name;

//glm
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>

//glu
#include<GL/GLU.h>

//glfw
#include<GLFW/glfw3.h>


//xml
#include<pugixml.hpp>
typedef pugi::xml_node Node;


//cuda
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_functions.h>
#include<device_launch_parameters.h>

//curand
#include<curand.h>
#include<curand_kernel.h>


//arrayfire
#include<arrayfire.h>

//opencv
#include<opencv2/core.hpp>
#include<opencv2/core/cuda.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

//skeleton key
#include<skeleton_key.h>