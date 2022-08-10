#pragma once
#include"header_manifold.h"

ON_STRUCTURE Meta{

	ON_STRUCTURE Parameter{
		static std::string output_path = "C:/Users/Thelonious/source/repos/Omega null/Omega null/output/";
		static std::string topdir = "C:/Users/Thelonious/source/repos/Omega null/";
	};

	ON_PROCESS Debug{
		template<typename Text>
		static void print(Text text) {
		std::cout << std::endl << text << std::endl;
		}
	};

}