#pragma once
#include"header_manifold.h"

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