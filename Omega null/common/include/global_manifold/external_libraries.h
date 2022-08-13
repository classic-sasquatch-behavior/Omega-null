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

//filesystem
#include<filesystem>
namespace fs = std::filesystem;

//xml
#include<pugixml.hpp>
typedef pugi::xml_node Node;

//cuda
#include<cuda.h>

//arrayfire
#include<arrayfire.h>

//opencv
#include<opencv2/core.hpp>