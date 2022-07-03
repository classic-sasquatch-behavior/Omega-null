#include"../global_includes.h"




namespace fs = std::filesystem;
namespace on {
	ParseQueue::ParseQueue(std::string topdir) {
		_topdir = topdir;
	}

	void ParseQueue::parse() {
		find_on_files();
		enter_parse_queue();
	}

	void ParseQueue::find_on_files() {
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

	void ParseQueue::print_found_files() {
		std::cout << "found " << _parse_queue.size() << " .on files:" << std::endl;
		
		while (!_parse_queue.empty()) { //make queue macro later
			fs::path current_path = _parse_queue.front(); //might have to be converting path to string explicitly at some point? maybe? double slash is problematic, whats that about
			_parse_queue.pop();
			std::cout << current_path << std::endl;
		}
		std::cout << std::endl;
	}

	void ParseQueue::enter_parse_queue() {
		RUN_QUEUE(_parse_queue, current_path,
			read_on_file(current_path);
		);
	}

	void ParseQueue::read_on_file(fs::path path_to_input) {
		Lexer* reader = new Lexer();
		std::ifstream file(path_to_input);
		
		while(file.get(reader->pointer)){
			reader->read(reader->pointer);
		}
	}
}





