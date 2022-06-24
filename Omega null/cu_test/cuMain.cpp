#include"global_includes.h"

std::string topdir = "C:/Users/Thelonious/source/repos/Omega null/Omega null/cu_test";


int main() {
	
	on::Parser* parser = new on::Parser(topdir);
	parser->parse();

	return 0;
}