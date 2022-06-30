#include"../../global_includes.h"



namespace on {



	Parser::Parser() {
		initialize_page();
	}

	void Parser::initialize_page() {
		Structure* page = new Structure(page, 0);
		_structures.push_back(page);
		_current_structure = page;
		_buffer = "page";
		flush_buffer(1);
	}

	//-1 for flush without push, 0 for raw text, 1 to begin structure, 2 to end structure, 3 to identify arguments
	std::string Parser::flush_buffer(int argument) {
		std::string begin = "begin ";
		std::string end = "end ";

		switch (argument) {
		case 0: _content.push_back(_buffer); break;
		case 1: begin += _buffer; _content.push_back(begin); break;
		case 2: end += _buffer; _content.push_back(end); break;
		default: std::cout << "ERROR - flush_buffer argument undefined: " << argument << std::endl; break;
		}
		std::string result = _buffer;
		_buffer.clear();
		return result;
	}

#pragma region states

	void Parser::step(char input) {
		switch (state) {
		case 0:
			state_0(input); break;
		case 1:
			state_1(input); break;
		case 2:
			state_2(input); break;
		default:
			std::cout << "ERROR - undefined state: " << input << std::endl; break;
		}
	}

	void Parser::state_0(char input) { //idle
		switch (input) {
		case '<': change_state(2); break;
		case ' ': break;
		case '\n': break;
		default: change_state(1); _buffer += input; break;
		}
	}

	void Parser::state_1(char input) { //raw c
		switch (input) {
		case '\n': flush_buffer(0); break;
		default: _buffer += input;  break;
		}
	}

	void Parser::state_2(char input) { //header
		switch (substate) {
		case 0: state_2_substate_0(input); break;
		case 1: state_2_substate_1(input); break;
		case 2: state_2_substate_2(input); break;
		case 3: state_2_substate_3(input); break;
		case 4: state_2_substate_4(input); break;
		default: std::cout << "ERROR - substate undefined: " << substate << std::endl; break;
		}
	}

	void Parser::state_2_substate_0(char input) { //structure type
		switch (input) {
		case '<': break;
		case ' ': break; //needs to be ignored before type, and serve as end of type after
		case '/': change_substate(3); break;
		case ',': //create structure and set type, set dims if you have them, and go to arguments.
		case '(': change_substate(4); break;
		case '>': flush_buffer(1); change_state(0); break;
		default: _buffer += input; break;
		}
	}

	void Parser::state_2_substate_1(char input) { //structure name
		switch (input) {
		case ' ': break;
		case ',': //set name of structure
		case '>': flush_buffer(1); change_state(0); break;
		default: _buffer += input; break;
		}
	}

	void Parser::state_2_substate_2(char input) { //arguments
		switch (input) {
		case ' ': break;
		case '=': //flush_buffer(-1) and do something with the argument
		case '{': //understand that we are specifically beginning data argument
		case ',': //record argument value and prepare to hear a new one
		case '>': change_state(0); //close out any active arguments if applicable
		default: break;
		}
	}

	void Parser::state_2_substate_3(char input) { //end tag
		switch (input) {
		case ' ': break;
		case '\n': flush_buffer(2); break;
		defualt: _buffer += input; break;
		}
	}

	void Parser::state_2_substate_4(char input) { //define dims
		switch (input) {

		}
	}


#pragma endregion


	void Parser::change_state(int new_state) {
		int& old_state = state;
		switch (new_state) {
		case 0:
		case 1:
		case 2:
		default: std::cout << "ERROR - undefined new state: " << new_state << std::endl; break;

		}
		switch (old_state) {
		case 0:
		case 1:
		case 2:
		default: std::cout << "ERROR - undefined old state: " << old_state << std::endl; break;
		}
	}

	void Parser::change_substate(int new_substate) {
		int old_substate = substate;
		switch (new_substate) {
		case 0: substate = 0; break;
		case 1: substate = 1; break;
		case 2: substate = 2; break;
		case 3: substate = 3; break;
		case 4: substate = 4; break;
		default: std::cout << "ERROR - undefined new substate: " << new_substate << std::endl;
		}
		switch (old_substate) {
		case 0: 
		case 1:
		case 2:
		case 3:
		case 4:
		default: std::cout << "ERROR - undefined old substate: " << old_substate << std::endl;
		}
	}


}