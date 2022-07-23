#include"headers/global_includes.h"




int main() {
	
	on::Fetcher* on_queue = new on::Fetcher(topdir);
	on_queue->start();

	return 0;
}