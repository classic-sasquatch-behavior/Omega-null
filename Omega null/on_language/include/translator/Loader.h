#pragma once
#include"language_manifold.h"
#include"omega_null.h"



namespace on {

	On_Structure Loader{
	On_Being File{
		fs::path source;
		std::fstream stream;
		std::string extension;

		File(fs::path input) {
			source = input;
			stream.open(source.c_str());
			extension = source.extension().generic_string();
		}

	};

	On_Process Load {
		static void structures(std::queue<fs::path> &file_queue);

		static void on_file(fs::path current_path, std::vector<Node>& kernels);

		static void on_embedded(File& current_file, std::vector<Node>& kernels);

		static Node xml(std::string document);

		static void project(std::string topdir, std::queue<fs::path>& file_queue);

		static std::ofstream file(std::string file_name, std::string extension);

		static KernelData kernel_data(Node root);
	};

	On_Structure LoadStructures{
		static std::queue<fs::path> file_queue;
	}

	}
}
