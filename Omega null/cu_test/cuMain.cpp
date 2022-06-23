#include"host_headers.h"
#include"device_headers.h"
#include"on_structures.h"
#include"classes.h"




int main() {
	
	on::Parser* parser = new on::Parser();
	parser->parse();

	return 0;
}