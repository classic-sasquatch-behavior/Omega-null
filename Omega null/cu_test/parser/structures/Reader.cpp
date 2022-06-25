#include"../../global_includes.h"







Reader::Reader() {

}



void Reader::step(char input) {
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



#pragma region states

void Reader::state_0(char input) { //idle
	switch (input) {
	case '<':
	case ' ':
	case '\n':
	default: break;
	}
}

void Reader::state_1(char input) { //raw c
	switch (input) {
	case '\n':
	default: break;
	}
}

void Reader::state_2(char input) { //header
	switch (substate) {
	case 0:
		state_2_substate_0(input);
	case 1:
		state_2_substate_1(input);
	case 2:
		state_2_substate_2(input);
	default: std::cout << "ERROR - substate undefined: " << substate << std::endl; break;
	}
}

void Reader::state_2_substate_0(char input) { //structure type
	switch (input) {
	case ' ':
	case ',':
	case '(':
	case '>':
	default: break;
	}
}

void Reader::state_2_substate_1(char input) { //structure name
	switch (input) {
	case ' ':
	case ',':
	case '>':
	default: break;
	}
}

void Reader::state_2_substate_2(char input) { //arguments
	switch (input) {
	case ' ':
	case '=':
	case '{':
	case ',':
	case '>':
	default: break;
	}
}


#pragma endregion