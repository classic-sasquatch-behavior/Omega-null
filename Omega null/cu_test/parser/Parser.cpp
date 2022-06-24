#include"../global_includes.h"




namespace fs = std::filesystem;
namespace on {
	Parser::Parser(std::string topdir) {
		_topdir = topdir;
	}

	void Parser::parse() {
		find_on_files();
		//print_found_files();














	}

	void Parser::find_on_files() {
		fs::path src_path = _topdir;
		if (fs::exists(src_path) && fs::is_directory(src_path)) {
			for (const auto& entry : fs::recursive_directory_iterator(src_path)) {
				fs::path current_entry = entry.path();
				if (current_entry.extension() == ".on") {
					add_path_to_queue(current_entry);
				}
			}
		}
	}

	void Parser::print_found_files() {
		std::cout << "found " << _queue.size() << " .on files:" << std::endl;
		
		while (!_queue.empty()) { //make queue macro later
			fs::path current_path = _queue.front(); //might have to be converting path to string explicitly at some point? maybe? double slash is problematic, whats that about
			_queue.pop();
			std::cout << current_path << std::endl;
		}
		std::cout << std::endl;
	}
















}





