#pragma once


#define RUN_QUEUE(queue_name, iterator_name, content) \
while(!queue_name.empty()){ \
	auto iterator_name = queue_name.front(); \
	queue_name.pop(); \
	content;}

#define __THROW_ERROR(function_name, reason_for_error, data_to_display) \
std::cout << "ERROR in function_name - reason_for_error: " << data_to_display << std::endl;



enum States {
	IDLE = 0, 
	RAW_TEXT = 1, 
	OPEN_TAG = 2, 
	STRUCTURE_TYPE = 3, 
	STRUCTURE_NAME = 4, 
	STRUCTURE_DIMS = 5, 
	STRUCTURE_DATA = 6, 
	CLOSE_TAG = 7,
	TAG_IDLE = 8
};

enum BufferArgs {
	RETURN = -1, 
	END_RAW_TEXT = 0, 
	MARK_BEGINNING_OF_STRUCTURE = 1,
	MARK_END_OF_STRUCTURE = 2,
	PUSH_BACK_DIM = 3
};





