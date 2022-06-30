#include"global_includes.h"

std::string topdir = "C:/Users/Thelonious/source/repos/Omega null/Omega null/cu_test";


int main() {
	
	on::ParseQueue* parser = new on::ParseQueue(topdir);
	parser->parse();

	return 0;
}