#include"global_includes.h"




int main() {
	
	on::ParseQueue* parser = new on::ParseQueue(topdir);
	parser->parse();

	return 0;
}