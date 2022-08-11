#pragma once
#include"header_manifold.h"

ON_STRUCTURE Loader{
	ON_PROCESS Load {
		static void structures(std::queue<fs::path> &file_queue);

		static Node xml(std::string document);

		static void project(std::string topdir, std::queue<fs::path>&file_queue);

		static std::ofstream file(std::string file_name, std::string extension);

		static KernelData kernel_data(Node root);
	};

	ON_STRUCTURE LoadStructures{
	static std::queue<fs::path> file_queue;
	}

}