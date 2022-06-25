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
	case 3:
		state_3(input); break;
	case 4:
		state_4(input); break;
	case 5:
		state_5(input); break;
	default:
		std::cout << "ERROR - undefined state: " << input << std::endl; break;
	}
}



#pragma region states

void state_0(char input) {
	switch (input) {

	}
}

void state_1(char input) {
	switch (input) {

	}
}

void state_2(char input) {
	switch (input) {

	}
}

void state_3(char input) {
	switch (input) {

	}
}

void state_4(char input) {
	switch (input) {

	}
}

void state_5(char input) {
	switch (input) {

	}
}

#pragma endregion