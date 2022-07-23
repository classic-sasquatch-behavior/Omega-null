#pragma once
#include"../../global_includes.h"


namespace on {
	struct Structure;


	struct Parser {
	public:
		Parser();
		~Parser();

#pragma region reader functions
		void read(char input);
		inline void step_up() { level++; }
		inline void step_down() { level--; }
		std::string flush_buffer(int argument);
		void change_state(int new_state);
		void initialize_page();
		void initialize_structure();
		void generate_report();
		void close_tag();
		void complete_type();
		void complete_name();
		void complete_dims();
		void complete_data();
		void complete_parent(std::string data_to_check);
		void wrap_up();
#pragma endregion

#pragma region writer functions
		void generate_code();
		void tokenize_content(std::vector<Structure*>& kernels);
		void identify_kernels();
		void initialize_files(std::string kernel_name, std::ofstream &header_file, std::ofstream &cuda_file);
		Structure* find_structure_with_name(std::string target_name);
		void generate_header(Structure* kernel, std::ofstream& header_file);
		void generate_cuda_code(Structure* kernel, std::ofstream& cuda_file);

		//string templates
		std::string template_kernel_begin(Structure* kernel);
		std::string template_kernel_content(Structure* kernel);
		std::string template_kernel_end(Structure* kernel);

		std::string template_launch_begin(Structure* kernel);
		std::string template_launch_dims(Structure* kernel);
		std::string template_launch_kernel_call(Structure* kernel);
		std::string template_launch_end(Structure* kernel);
#pragma endregion

#pragma region public data
		int state = 0;
		int level = 0;
		char pointer;
#pragma endregion

#pragma region states
		void idle_state(char input); 
		void raw_text_state(char input);
		void open_tag_state(char input);
		void structure_type_state(char input);
		void structure_name_state(char input);
		void structure_dims_state(char input);
		void structure_data_state(char input);
		void close_tag_state(char input);
		void tag_idle_state(char input);
#pragma endregion

#pragma region get-set
		std::string get_content(int index) { return _content[index]; }
		std::string get_buffer() { return _buffer; }
		Structure* get_structure(int index) { return _structures[index]; }
		Structure* get_current_structure() { return _current_structure; }

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