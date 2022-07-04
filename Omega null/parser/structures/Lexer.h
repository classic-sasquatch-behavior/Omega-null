#pragma once
#include"../../global_includes.h"


namespace on {
	struct Structure;


	struct Lexer {
	public:
		Lexer();
		~Lexer();

		void read(char input);
		inline void step_up() { level++; }
		inline void step_down() { level--; }
		std::string flush_buffer(int argument);
		void change_state(int new_state);
		void initialize_page();
		void initialize_structure();
		void generate_report();
		
		
		void complete_type();
		void complete_name();
		void complete_dims();
		void complete_data();

		int state = 0;
		int level = 0;
		char pointer;


		//states
		void idle_state(char input); 
		void raw_text_state(char input);
		void open_tag_state(char input);
		void structure_type_state(char input);
		void structure_name_state(char input);
		void structure_dims_state(char input);
		void structure_data_state(char input);
		void close_tag_state(char input);
		void tag_idle_state(char input);






#pragma region get-set
		std::string get_content(int index) { return _content[index]; }
		std::string get_buffer() { return _buffer; }
		Structure* get_structure(int index) { return _structures[index]; }
		Structure* get_current_structure() { return _current_structure; }

		//TODO: fill in parser get-set
		void add_content(std::string input) { _content.push_back(input); }
		void add_to_buffer(char input) { _buffer += input; }
		void add_structure(Structure* input) { _structures.push_back(input); }
		void set_current_structure(Structure* input) { _current_structure = input; }
#pragma endregion

	private:
		std::vector<std::string> _content;
		std::string _buffer;
		std::vector<Structure*> _structures;
		Structure* _current_structure;
	};

}