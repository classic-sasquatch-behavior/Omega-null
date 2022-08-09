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

	ON_PROCESS Write{

		static void launcher_declaration(KernelData kernel, std::ofstream & header) {

			std::string name = kernel.name;
			std::string shape = kernel.shape;
			std::string data = kernel.data;
			uint block_dim_x = kernel.block_dim_x();
			uint block_dim_y = kernel.block_dim_y();

			header << "void " << name << "_launch(" << data << ");" << std::endl;
		}

		static void to_kernel(std::string content, std::ofstream & cuda_file) {
			cuda_file << content << std::endl;
		}

		static void kernel_declaration(Node node, KernelData kernel, std::ofstream & cuda_file) {
			std::string name = kernel.name;
			std::string dims = kernel.dims;
			std::string shape = kernel.shape;
			std::string data = kernel.data;

			cuda_file << "__global__ void " << name << "(" << data << "){" << std::endl;
			cuda_file << "GET_DIMS(" << dims << ");" << std::endl;
			cuda_file << "CHECK_BOUNDS(" << shape << ".maj_span, " << shape << ".min_span);" << std::endl;
		}

		static void for_element(Node node, KernelData data, std::ofstream & cuda_file) {

		}

		static void for_neighbor(Node node, KernelData data, std::ofstream & cuda_file) {

		}

		static void for_maj(Node node, KernelData data, std::ofstream & cuda_file) {

		}

		static void for_min(Node node, KernelData data, std::ofstream & cuda_file) {

		}

		static void cast_down(Node node, KernelData data, std::ofstream & cuda_file) {

		}

		static void cast_up(Node node, KernelData data, std::ofstream & cuda_file) {

		}

		static void launcher_definition(KernelData kernel, std::ofstream & cuda_file) {

			std::string name = kernel.name;
			std::string shape = kernel.shape;
			std::string data = kernel.data;
			std::string block_dim_x = std::to_string(kernel.block_dim_x());
			std::string block_dim_y = std::to_string(kernel.block_dim_y());

			cuda_file << "\n"
				"void " + name + "_launch(" + data + ")" + "\n"
				"on::Tensor& shape = " + shape + ";" + "\n"
				"\n"
				"unsigned int block_dim_x = " + block_dim_x + ";" + "\n"
				"unsigned int block_dim_y = " + block_dim_y + ";" + "\n"
				"unsigned int grid_dim_x = (shape.maj_span - (shape.maj_span % block_dim_x))/block_dim_x;" + "\n"
				"unsigned int grid_dim_y = (shape.min_span - (shape.min_span % block_dim_y))/block_dim_y;" + "\n"
				"dim3 num_blocks(grid_dim_x + 1, grid_dim_y + 1);" + "\n"
				"dim3 threads_per_block(block_dim_x, block_dim_y)" + "\n"
				+ name + "<<<num_block, threads_per_block>>>(" + data + ");" + "\n"
				"}" + "\n";
		}

	};

	ON_BEING WriteData{
		WriteData() {}
		int layer = 0;
		//current layer
	};

}

ON_STRUCTURE Loader{
	ON_PROCESS Load{
		static void structures() {

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

ON_STRUCTURE Reader{
	using ON_STRUCTURE Meta;
	using ON_STRUCTURE Writer;
	ON_PROCESS Read{
		static void next_node(Node root, KernelData data, std::ofstream & cuda_file) {
		Debug::print("traversing node");
		std::string structure_type = root.name();

		if (structure_type == "Kernel") { Write::begin_kernel(root, data, cuda_file); }
		if (structure_type == "For_Element") { Write::for_element(root, data, cuda_file); }
		if (structure_type == "For_Neighbor") { Write::for_neighbor(root, data, cuda_file); }
		if (structure_type == "For_Maj") { Write::for_maj(root, data, cuda_file); }
		if (structure_type == "For_Min") { Write::for_min(root, data, cuda_file); }
		if (structure_type == "Cast_Down") { Write::cast_down(root, data, cuda_file); }
		if (structure_type == "Cast_Up") { Write::cast_up(root, data, cuda_file); }

		for (Node node : root) {
			if (node.type() == pugi::node_pcdata) { Write::write_to_kernel(node.text().as_string(), cuda_file); }
			else { Read::next_node(node, data, cuda_file); }
		}

		cuda_file << std::endl << "}" << std::endl;
		}
	};
};

using ON_STRUCTURE Meta;
using ON_STRUCTURE Writer;
using ON_STRUCTURE Loader;
using ON_STRUCTURE Reader;
int main() {

	Load::project(Parameter::topdir, LoadStructures::file_queue);




	Debug::print("making .h and .cu for each .on file");
	//2) for each .on file, make a .h and a .cu file with the same name
	while(!LoadStructures::file_queue.empty()) {
		auto current_path = LoadStructures::file_queue.front();
		LoadStructures::file_queue.pop();

		pugi::xml_document on_file;
		pugi::xml_parse_result loaded_file_successfully = on_file.load_file(current_path.c_str());
		if (!loaded_file_successfully) { std::cout << "XML file " << current_path.c_str() << " could not be loaded." << std::endl; }

		std::string file_name = Parameter::output_path + current_path.stem().u8string();
		Debug::print(file_name);
		std::ofstream header_stream = Load::file(file_name, ".h");
		std::ofstream cuda_stream = Load::file(file_name, ".cu");







		//3) for each kernel in .on file:
		//	a) use kernel header to write launch function
		//	b) use whole kernel structure to write kernel in .cu 
		for (Node root : on_file) {
			Debug::print("making kernel");
			KernelData kernel = Load::kernel_data(root);
			Write::launcher_declaration(kernel, header_stream);
			Read::next_node(root, kernel, cuda_stream);
			Write::launcher_definition(kernel, cuda_stream);
		}







	}

	//4) find and replace #include"name.on" with #include"name.h"

	return 0;
}