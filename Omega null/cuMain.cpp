#include"global_includes.h"

std::string topdir = "C:/Users/Thelonious/source/repos/Omega null";


int main() {
	
	on::ParseQueue* parser = new on::ParseQueue(topdir);
	parser->parse();

	return 0;
}