#include"manifold.h"


const std::string output_path = "C:/Users/Thelonious/source/repos/Omega null/Omega null/output/";
const std::string topdir = "C:/Users/Thelonious/source/repos/Omega null/";

struct KernelData {
	KernelData(){}
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

std::ofstream open_file(std::string file_name, std::string extension) {
	std::string full_name = file_name + extension;
	std::ofstream file;
	file.open(output_path + full_name);
	return file;
}

void close_file(std::ofstream& file) {
	file.close();
}

KernelData load_kernel(Node root) {
	KernelData kernel;
	kernel.name = root.attribute("name").as_string();
	kernel.data = root.attribute("data").as_string();
	kernel.dims = root.attribute("dims").as_string();
	kernel.shape = root.attribute("shape").as_string();
	return kernel;
}







#pragma region writer functions

void write_launch_function(KernelData kernel, std::ofstream& header, std::ofstream& cuda_file) {

	std::string name = kernel.name;
	std::string shape = kernel.shape;
	std::string data = kernel.data;
	uint block_dim_x = kernel.block_dim_x();
	uint block_dim_y = kernel.block_dim_y();

	header << "void " << name << "_launch(" << data << ");" << std::endl;

	cuda_file << "void " << name << "_launch(" << data << "){" << std::endl;
	cuda_file << "on::Tensor& shape = " << shape << ";" << std::endl;
	cuda_file << "unsigned int block_dim_x = " << block_dim_x << ";" << std::endl;
	cuda_file << "unsigned int block_dim_y = " << block_dim_y << ";" << std::endl;
	cuda_file << "unsigned int grid_dim_x = (shape.maj_span - (shape.maj_span % block_dim_x))/block_dim_x;" << std::endl;
	cuda_file << "unsigned int grid_dim_y = (shape.min_span - (shape.min_span % block_dim_y))/block_dim_y;" << std::endl;
	cuda_file << "dim3 num_blocks(grid_dim_x + 1, grid_dim_y + 1);" << std::endl;
	cuda_file << "dim3 threads_per_block(block_dim_x, block_dim_y)" << std::endl;
	cuda_file << name << "<<<num_block, threads_per_block>>>(" << data << ");" << std::endl;
	cuda_file << "}" << std::endl;
}


void write_to_kernel(std::string content, std::ofstream& cuda_file) {
	cuda_file << content << std::endl;
}

void begin_kernel(Node node, KernelData kernel, std::ofstream& cuda_file) {
	std::string name = kernel.name;
	std::string dims = kernel.dims;
	std::string shape = kernel.shape;
	std::string data = kernel.data;

	cuda_file << "__global__ void " << name << "(" << data << "){" << std::endl;
	cuda_file << "GET_DIMS(" << dims << ");" << std::endl;
	cuda_file << "CHECK_BOUNDS(" << shape << ".maj_span, " << shape << ".min_span);" << std::endl;
}

void for_element(Node node, KernelData data, std::ofstream& cuda_file) {

}

void for_neighbor(Node node, KernelData data, std::ofstream& cuda_file) {

}

void for_maj(Node node, KernelData data, std::ofstream& cuda_file) {

}

void for_min(Node node, KernelData data, std::ofstream& cuda_file) {

}

void cast_down(Node node, KernelData data, std::ofstream& cuda_file) {

}

void cast_up(Node node, KernelData data, std::ofstream& cuda_file) {

}

#pragma endregion




void traverse_node(Node root, KernelData data, std::ofstream& cuda_file) {
	std::string structure_type = root.name();

	if (structure_type == "Kernel") { begin_kernel(root, data, cuda_file); }
	if (structure_type == "For_Element") { for_element(root, data, cuda_file); }
	if (structure_type == "For_Neighbor") { for_neighbor(root, data, cuda_file); }
	if (structure_type == "For_Maj") { for_maj(root, data, cuda_file); }
	if (structure_type == "For_Min") { for_min(root, data, cuda_file); }
	if (structure_type == "Cast_Down") { cast_down(root, data, cuda_file); }
	if (structure_type == "Cast_Up") { cast_up(root, data, cuda_file); }
	
	for (Node node : root) {
		if (node.type() == pugi::node_pcdata) { write_to_kernel(node.text().as_string(), cuda_file); }
		else { traverse_node(node, data, cuda_file); }
	}

	cuda_file << std::endl << "}" << std::endl;
}



int main() {
	
	std::queue<fs::path> file_queue;

	//1) get all .on files in project dir
	fs::path src_path = topdir;
	if (fs::exists(src_path) && fs::is_directory(src_path)) {
		for (const auto& entry : fs::recursive_directory_iterator(src_path)) {
			fs::path current_path = entry.path();
			if (current_path.extension() == ".on") {
				file_queue.push(current_path);
			}
		}
	}

	//2) for each .on file, make a .h and a .cu file with the same name
	while(!file_queue.empty()) {
		auto current_path = file_queue.front();
		file_queue.pop();

		pugi::xml_document on_file;
		pugi::xml_parse_result loaded_file_successfully = on_file.load_file(current_path.c_str());
		if (!loaded_file_successfully) { std::cout << "XML file " << current_path.c_str() << " could not be loaded." << std::endl; }

		std::string file_name = current_path.filename().u8string();
		std::ofstream header_stream = open_file(file_name, ".h");
		std::ofstream cuda_stream = open_file(file_name, ".cu");


		//3) for each kernel in .on file:
		//	a) use kernel header to write launch function
		//	b) use whole kernel structure to write kernel in .cu 
		for (Node root : on_file) {
			KernelData kernel = load_kernel(root);
			write_launch_function(kernel, header_stream, cuda_stream);
			traverse_node(root, kernel, cuda_stream);
		}
	}

	//4) find and replace #include"name.on" with #include"name.h"

	return 0;
}