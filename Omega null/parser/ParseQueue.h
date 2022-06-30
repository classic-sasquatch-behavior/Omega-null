#pragma once
#include"../global_includes.h"


namespace fs = std::filesystem;
namespace on{



class ParseQueue {
public:
	ParseQueue(std::string topdir);
	~ParseQueue();

	void parse();
	void find_on_files();
	void enter_parse_queue();
	void read_on_file(fs::path path_to_input);

#pragma region util
	void print_found_files();

#pragma endregion

#pragma region get/set


	inline void add_path_to_queue(fs::path input) { _parse_queue.push(input); }

#pragma endregion


private:
	std::string _topdir;
	std::queue<fs::path> _parse_queue;


};






}