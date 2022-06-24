#pragma once


#define Run_Queue(queue_name, iterator_name, content) \
while(!queue_name.empty()){ \
	auto iterator_name = queue_name.front(); \
	queue_name.pop(); \
	content;}

