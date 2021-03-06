#include"../../headers/global_includes.h"




namespace fs = std::filesystem;
namespace on {
	Fetcher::Fetcher(std::string topdir) {
		_topdir = topdir;
	}

	void Fetcher::parse() {
		find_on_files();
		enter_parse_queue();
	}

	void Fetcher::find_on_files() {
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

	void Fetcher::print_found_files() {
		std::cout << "found " << _parse_queue.size() << " .on files:" << std::endl;
		
		while (!_parse_queue.empty()) { //make queue macro later
			fs::path current_path = _parse_queue.front(); //might have to be converting path to string explicitly at some point? maybe? double slash is problematic, whats that about
			_parse_queue.pop();
			std::cout << current_path << std::endl;
		}
		std::cout << std::endl;
	}

	void Fetcher::enter_parse_queue() {
		RUN_QUEUE(_parse_queue, current_path,
			read_on_file(current_path);
		);
	}







	void Fetcher::read_on_file(fs::path path_to_input) {
		Parser* reader = new Parser();
		std::ifstream file(path_to_input);
		std::cout << "reading .on file..." << std::endl;


		while(file.get(reader->pointer)){
			reader->read(reader->pointer);
		}
		reader->wrap_up();
		reader->generate_report(); //for debug
		reader->generate_code();
	}
}





