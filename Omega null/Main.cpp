#include"manifold.h"


const std::string output_path = "C:/Users/Thelonious/source/repos/Omega null/Omega null/output/";
const std::string topdir = "C:/Users/Thelonious/source/repos/Omega null/";


std::string initialize_header(std::string file_name) {

}

std::string initialize_cuda_file(std::string file_name) {

}

void write_header(std::string content, std::string path_to_header) {

}

void write_cuda(std::string content, std::string path_to_cuda) {

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

	while(!file_queue.empty()) {
		auto current_path = file_queue.front();
		file_queue.pop();

		pugi::xml_document on_file;
		pugi::xml_parse_result loaded_file_successfully = on_file.load_file(current_path.c_str());
		if (!loaded_file_successfully) { std::cout << "XML file " << current_path.c_str() << " could not be loaded." << std::endl; }

		std::string file_name = current_path.filename().u8string();
		std::string header_content = initialize_header(file_name);
		std::string cuda_content = initialize_cuda_file(file_name);

		for (Node Kernel : on_file) {


		}
	}





	//2) for each .on file, make a .h and a .cu file with the same name
		std::string header_name = kernel_name + ".h";
		std::string cuda_file_name = kernel_name + ".cu";

		header_file.open(output_path + header_name);
		cuda_file.open(output_path + cuda_file_name);

	//3) for each kernel in .on file:
	//	a) use kernel header to write launch function
	//	b) use whole kernel structure to write kernel in .cu
		std::vector<Structure*> kernels;
		tokenize_content(kernels);
		if (kernels.size() <= 0) { __THROW_ERROR(generate code, no kernels found, "very sad"); }

		std::ofstream header_file;
		std::ofstream cuda_file;
		std::string kernel_name = kernels[0]->get_name();
		initialize_files(kernel_name, header_file, cuda_file);

		for (Structure* kernel : kernels) {
			generate_header(kernel, header_file);
			generate_cuda_code(kernel, cuda_file);
		}

		header_file.close();
		cuda_file.close();


	//4) find and replace #include"name.on" with #include"name.h"







	return 0;
}