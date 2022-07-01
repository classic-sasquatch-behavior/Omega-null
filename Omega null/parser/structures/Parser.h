#pragma once
#include"../../global_includes.h"


namespace on {

	struct Parser {
	public:
		Parser();
		~Parser();

		void step(char input);
		std::string flush_buffer(int argument);
		void change_state(int new_state);
		void change_substate(int new_substate);
		void initialize_page();
		void initialize_structure(Structure* parent, int level);

		int state = 0;
		int substate = 0;
		int level = 0;
		char pointer;

		void state_0(char input);
		void state_1(char input);
		void state_2(char input);
		void state_2_substate_0(char input);
		void state_2_substate_1(char input);
		void state_2_substate_2(char input);
		void state_2_substate_3(char input);
		void state_2_substate_4(char input);

#pragma region get-set



		//TODO: fill in parser get-set








#pragma endregion

	private:
		std::vector<std::string> _content;
		std::string _buffer;
		std::vector<Structure*> _structures;
		Structure* _current_structure;
	};

}