#pragma once


//filestreams
#include<iostream>
#include<fstream>
#include<sstream>
using std::endl;
typedef std::ofstream Filestream;

//standard lib
#include<queue>
#include<string>

//TODO find a new way to navigate the filesystem
//filesystem
#include<filesystem>
namespace fs = std::filesystem;

//xml
#include<pugixml.hpp>
typedef pugi::xml_node Node;

//cuda
#include<cuda.h>
#include<cuda_runtime_api.h>

//arrayfire
#include<arrayfire.h>

//opencv
#include<opencv2/core.hpp>
#include<opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>