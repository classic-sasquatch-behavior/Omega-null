#include"manifold.h"





ON_STRUCTURE Loader{
	using ON_STRUCTURE Meta;
	using ON_STRUCTURE Reader;
	using ON_STRUCTURE Writer;

	ON_BEING File{
		fs::path source;
		std::fstream stream;
		std::string extension;

		File(fs::path input) {
			source = input;
			stream.open(source.c_str());
			extension = source.extension().generic_string();
		}

	};

	

	void Load::project(std::string topdir, std::queue<fs::path>&file_queue) {
		fs::path src_path = topdir;
		if (fs::exists(src_path) && fs::is_directory(src_path)) {

			for (const auto& entry : fs::recursive_directory_iterator(src_path)) {
				fs::path current_path = entry.path();

				if (current_path.extension() == ".on") {
					file_queue.push(current_path);
				}
				else if (auto ext = current_path.extension(); (Meta::Configuration::ON_EMBEDDED) 
					&& ((ext == ".h") || (ext == ".cpp") || (ext == ".cu"))) {
					file_queue.push(current_path);
				}
			}

		}
		else { Debug::print("topdir invalid"); }
	}


	Node Load::xml(std::string document) {
		pugi::xml_document root;
		root.load_string(document.c_str());
		return root;
	}

	void Load::structures(std::queue<fs::path>& file_queue) {
		while (!file_queue.empty()) {
			auto current_path = file_queue.front();
			file_queue.pop();

			File current_file(current_path);
			std::vector<Node> kernels;

			if (Configuration::ON_EMBEDDED) {
				std::string line;
				std::string kernel_buffer;

				bool reading_kernel = false;
				while (std::getline(current_file.stream, line) ) {

					if (!reading_kernel) {

						if (line == "__begin_kernel__") {
							reading_kernel = true;
						}
						else {
							continue;
						}
					}

					else if (reading_kernel) {

						if (line == "__end_kernel__") { //probably wont work, and definitely wont be robust even if it does. find solution using regex.
							reading_kernel = false;
							kernels.push_back(Load::xml(kernel_buffer));
							kernel_buffer.clear();
						}
						else {
							kernel_buffer += line;
						}
					}
					
				}
			} 

			else if (Configuration::ON_FILE) {
				pugi::xml_document on_file;
				on_file.load_file(current_path.c_str());
				kernels.push_back(on_file);
			}

			//initialize header and cuda file (if configured for embedded statements, only create one for the whole file). always runs
			std::string file_name = Parameter::output_path + current_path.stem().u8string();
			Filestream header = Load::file(file_name + "_launch", ".h");
			Filestream cuda = Load::file(file_name + "_kernels", ".cu");

			for (Node root : kernels) {








			}









			//traverse actual xml structure. always runs. in FILE mode it looks at a file at a time, in EMBEDDED mode it looks at a list of embedded structures
			Node root = on_file;
			if (root.first_child().name() != "Kernel") root = root.first_child();

			for (Node node : root) {
				if (node.type() == pugi::node_pcdata) { Write::text_to_kernel(cuda, node.text().as_string()); continue; }
				KernelData data = Load::kernel_data(node);
				Write::launcher_declaration(data, header);
				Read::next_node(node, data, cuda);
				Write::launcher_definition(data, cuda);
			}









		}
	}






	std::ofstream Load::file(std::string file_name, std::string extension) {
		std::string full_name = file_name + extension;
		std::ofstream file;
		file.open(full_name);
		file << std::endl;
		return file;
	}

	KernelData Load::kernel_data(Node root) {
		KernelData kernel;
		kernel.name = root.attribute("name").as_string();
		kernel.data = root.attribute("data").as_string();
		kernel.dims = root.attribute("dims").as_string();
		kernel.shape = root.attribute("shape").as_string();
		return kernel;
	}
}