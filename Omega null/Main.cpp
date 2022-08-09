#include"manifold.h"

/*
	had to include this to the .vcxproj:

  <ItemGroup>
	<ClInclude Include="output\*.h" />
	<CudaCompile Include="output\*.cu" />
  </ItemGroup>

  but, it doesn't work as automatically as I'd like it to. we'll do it manually for now and 
  figure out how to make this work later.
*/

ON_BEING KernelData{

	KernelData() {}
	int layer = 0;

	std::string name;
	std::string dims;
	std::string shape;
	std::string data;

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

		static void text_to_kernel(KernelData& data, Filestream& cuda_file, std::string content) {
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

ON_STRUCTURE Reader{
	using ON_STRUCTURE Meta;
	using ON_STRUCTURE Writer;

	ON_PROCESS Read{
		static void next_node(Node root, KernelData data, std::ofstream & cuda_file) {

		std::string structure_type = root.name();

		cuda_file << Get::tabs(data.layer);
		if (structure_type == "Kernel") { Write::kernel_declaration(root, data, cuda_file); }
		if (structure_type == "For_Element") { Write::for_element(root, data, cuda_file); }
		if (structure_type == "For_Neighbor") { Write::for_neighbor(root, data, cuda_file); }
		if (structure_type == "For_Maj") { Write::for_maj(root, data, cuda_file); }
		if (structure_type == "For_Min") { Write::for_min(root, data, cuda_file); }
		if (structure_type == "Cast_Down") { Write::cast_down(root, data, cuda_file); }
		if (structure_type == "Cast_Up") { Write::cast_up(root, data, cuda_file); }

		for (Node node : root) {
			if (node.type() == pugi::node_pcdata) { Write::text_to_kernel(data, cuda_file, node.text().as_string()); }
			else { Read::next_node(node, data, cuda_file); }
		}

		data.layer--;
		Write::to_kernel(data, cuda_file, "}\n");
		}
	};
};

ON_STRUCTURE Loader{
	using ON_STRUCTURE Meta;
	using ON_STRUCTURE Reader;
	using ON_STRUCTURE Writer;
	ON_PROCESS Load{
		static void structures(std::queue<fs::path> &file_queue) {
			while (!file_queue.empty()) {
				auto current_path = file_queue.front();
				file_queue.pop();

				pugi::xml_document on_file;
				pugi::xml_parse_result loaded_file_successfully = on_file.load_file(current_path.c_str());
				if (!loaded_file_successfully) { std::cout << "XML file " << current_path.c_str() << " could not be loaded." << std::endl; }

				std::string file_name = Parameter::output_path + current_path.stem().u8string();
				Filestream header = Load::file(file_name, ".h");
				Filestream cuda = Load::file(file_name, ".cu");

				for (Node kernel : on_file) {
					KernelData data = Load::kernel_data(kernel);
					Write::launcher_declaration(data, header);
					Read::next_node(kernel, data, cuda);
					Write::launcher_definition(data, cuda);
				}
			}
		}

		static void project(std::string topdir, std::queue<fs::path>&file_queue) {
			fs::path src_path = topdir;
			if (fs::exists(src_path) && fs::is_directory(src_path)) {
				for (const auto& entry : fs::recursive_directory_iterator(src_path)) {
					fs::path current_path = entry.path();
					if (current_path.extension() == ".on") {
						file_queue.push(current_path);
					}
				}
			}
		}

		static std::ofstream file(std::string file_name, std::string extension) {
			std::string full_name = file_name + extension;
			std::ofstream file;
			file.open(full_name);
			file << std::endl;
			return file;
		}

		static KernelData kernel_data(Node root) {
			KernelData kernel;
			kernel.name = root.attribute("name").as_string();
			kernel.data = root.attribute("data").as_string();
			kernel.dims = root.attribute("dims").as_string();
			kernel.shape = root.attribute("shape").as_string();
			return kernel;
		}
	};

	ON_STRUCTURE LoadStructures{
		static std::queue<fs::path> file_queue;
	}
}

using ON_STRUCTURE Meta;
using ON_STRUCTURE Writer;
using ON_STRUCTURE Loader;
using ON_STRUCTURE Reader;
int main() {

	Load::project(Parameter::topdir, LoadStructures::file_queue);
	Load::structures(LoadStructures::file_queue);

	//4) find and replace #include"name.on" with #include"name.h"

	return 0;
}