#pragma once
#include"header_manifold.h"

ON_STRUCTURE Writer{

	ON_PROCESS Get {
		static std::string tabs(int layer);
	};

	ON_PROCESS Write{

		static void launcher_declaration(KernelData & kernel, Filestream & header);

		static void to_kernel(KernelData& data, Filestream& cuda_file, std::string content);

		static void text_to_kernel(Filestream& cuda_file, std::string content);

		static void kernel_declaration(Node node, KernelData& kernel, Filestream& cuda_file);

		static void for_element(Node node, KernelData& data, Filestream& cuda_file);

		static void for_neighbor(Node node, KernelData& data, Filestream& cuda_file);

		static void for_maj(Node node, KernelData& data, Filestream& cuda_file);

		static void for_min(Node node, KernelData& data, Filestream& cuda_file);

		static void cast_down(Node node, KernelData& data, Filestream& cuda_file);

		static void cast_up(Node node, KernelData& data, Filestream& cuda_file);

		static void launcher_definition(KernelData & kernel, Filestream & cuda_file); 

	};

}