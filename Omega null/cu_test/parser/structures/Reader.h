#pragma once
#include"../../global_includes.h"



struct Reader {
public:
	Reader();
	~Reader();

	void step(char input);

	int state = 0;
	int substate = 0;
	int level = 0;
	char char_buffer;

	void state_0(char input);
	void state_1(char input);
	void state_2(char input);
	void state_2_substate_0(char input);
	void state_2_substate_1(char input);
	void state_2_substate_2(char input);

private:
	std::vector<std::string> _content;
	char _pointer;
};