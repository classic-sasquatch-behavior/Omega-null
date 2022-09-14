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

//TODO find a new way to navigate the filesystem
//filesystem
#include<filesystem>
namespace fs = std::filesystem;

//opengl
//#include<gl/GL.h>

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

#include<skeleton_key.h>