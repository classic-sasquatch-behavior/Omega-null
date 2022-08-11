#pragma once
#include"header_manifold.h"


ON_BEING KernelData{

	KernelData() {}
	int layer = 0;

	std::string name;
	std::string dims;
	std::string shape;
	std::string data;

	std::string raw_xml;

	std::string maj_dim;
	std::string min_dim;
	int num_dims() {
		int result = 0;

		return result;
	}

	int block_dim_x() {
		switch (num_dims()) {
		case 1: return 1024;
		case 2: return 32;
		default: return -1;
		}
	}
	int block_dim_y() {
		switch (num_dims()) {
		case 1: return 1;
		case 2:return 32;
		default: return -1;
		}
	}

};

