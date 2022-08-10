#pragma once
#include"header_manifold.h"

ON_STRUCTURE Writer{

	ON_PROCESS Get {
		static std::string tabs(int layer) {
			std::string result = "";
			for (int i = 0; i < layer; i++) {
				result += "\t";
			}
			return result;
		}
	};

	ON_PROCESS Write{

		static void launcher_declaration(KernelData & kernel, Filestream & header) {

			std::string name = kernel.name;
			std::string shape = kernel.shape;
			std::string data = kernel.data;
			uint block_dim_x = kernel.block_dim_x();
			uint block_dim_y = kernel.block_dim_y();

			header << "void " << name << "_launch(" << data << ");" << std::endl;
		}

		static void to_kernel( KernelData& data, Filestream & cuda_file, std::string content) {
			cuda_file << Get::tabs(data.layer) << content << std::endl;
		}

		static void text_to_kernel(Filestream& cuda_file, std::string content) {
			cuda_file << content;
		}

		static void kernel_declaration(Node node, KernelData& kernel, Filestream & cuda_file) {
			std::string name = kernel.name;
			std::string dims = kernel.dims;
			std::string shape = kernel.shape;
			std::string data = kernel.data;

			Write::to_kernel(kernel, cuda_file, "__global__ void " + name + "(" + data + "){"); kernel.layer++;
			Write::to_kernel(kernel, cuda_file, "GET_DIMS(" + dims + ");");
			Write::to_kernel(kernel, cuda_file, "CHECK_BOUNDS(" + shape + ".maj_span, " + shape + ".min_span);");
		}

		static void for_element(Node node, KernelData& data, Filestream & cuda_file) {

		}

		static void for_neighbor(Node node, KernelData& data, Filestream & cuda_file) {

		}

		static void for_maj(Node node, KernelData& data, Filestream & cuda_file) {

		}

		static void for_min(Node node, KernelData& data, Filestream & cuda_file) {

		}

		static void cast_down(Node node, KernelData& data, Filestream & cuda_file) {

		}

		static void cast_up(Node node, KernelData& data, Filestream & cuda_file) {

		}

		static void launcher_definition(KernelData& kernel, Filestream & cuda_file) {

			std::string name = kernel.name;
			std::string shape = kernel.shape;
			std::string data = kernel.data;
			std::string block_dim_x = std::to_string(kernel.block_dim_x());
			std::string block_dim_y = std::to_string(kernel.block_dim_y());

			Write::to_kernel(kernel, cuda_file, "void " + name + "_launch(" + data + "){");
			kernel.layer++;
			Write::to_kernel(kernel, cuda_file, "on::Tensor& shape = " + shape + ";");
			Write::to_kernel(kernel, cuda_file, "");
			Write::to_kernel(kernel, cuda_file, "unsigned int block_dim_x = " + block_dim_x + ";");
			Write::to_kernel(kernel, cuda_file, "unsigned int block_dim_y = " + block_dim_y + ";");
			Write::to_kernel(kernel, cuda_file, "unsigned int grid_dim_x = (shape.maj_span - (shape.maj_span % block_dim_x))/block_dim_x;");
			Write::to_kernel(kernel, cuda_file, "unsigned int grid_dim_y = (shape.min_span - (shape.min_span % block_dim_y))/block_dim_y;");
			Write::to_kernel(kernel, cuda_file, "");
			Write::to_kernel(kernel, cuda_file, "dim3 num_blocks(grid_dim_x + 1, grid_dim_y + 1);");
			Write::to_kernel(kernel, cuda_file, "dim3 threads_per_block(block_dim_x, block_dim_y)");
			Write::to_kernel(kernel, cuda_file, "");
			Write::to_kernel(kernel, cuda_file, name + "<<<num_block, threads_per_block>>>(" + data + ");");
			kernel.layer--;
			Write::to_kernel(kernel, cuda_file, "}"); 
		}

	};


}