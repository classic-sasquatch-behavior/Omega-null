#pragma once
#include"../global_includes.h"


namespace fs = std::filesystem;
namespace on{



class Parser {
public:
	Parser(std::string topdir);
	~Parser();

	void parse();
	void find_on_files();


#pragma region util
	void print_found_files();

#pragma endregion

#pragma region get/set


	inline void add_path_to_queue(fs::path input) { _queue.push(input); }

#pragma endregion


private:
	std::string _topdir;
	std::queue<fs::path> _queue;


};






}