#pragma once
#include"../global_includes.h"



namespace on{



class Parser {
public:
	Parser(std::string topdir);
	~Parser();

	void parse();
private:
	std::string _topdir;



};






}